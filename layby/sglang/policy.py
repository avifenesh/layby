"""Park eviction order for SGLang's radix tree (--radix-eviction-policy park), the port of
park/vllm/policy.py.

SGLang orders both tiers with one EvictionStrategy: the device heap holds GPU leaves, the host heap
holds nodes already evicted from the GPU (node.evicted), and get_priority(node) ranks a heap, smallest
first out. As in the vLLM adapter the GPU stays LRU and the host tier (the CPU tier the rule places
for) follows the predicted next use:

  hinted     a node the rule placed has a predicted next use: now + park_eta, or NEVER for park and
             drop (first out). The hint holds until the node is touched (a request matched it, read
             from node.last_access_time against the counter when the hint was set) or until it is
             GRACE seconds past due; then the node returns to LRU order.
  unhinted   next use estimated the LRU way: now + (time since last use).

Eviction takes the node predicted to be needed last. Times are the tick clock (rank 0's monotonic
time, broadcast), and a node's last use is the tick in which the logical counter passed its
last_access_time, so every TP rank computes the same order. A node shared with other sessions keeps
the earliest predicted use among the sessions that placed it.
"""
import bisect

from sglang.srt.mem_cache.evict_policy import EvictionStrategy
from sglang.srt.mem_cache.unified_cache.components import base as _counter_mod
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

from layby.sglang.state import GRACE, NEVER, STATE

FULL = ComponentType.FULL


def counter() -> float:
    """The next value of SGLang's logical access counter: a node touched after this read has
    last_access_time >= it (minus 1e-5 per tree level, see _match_post_processor)."""
    return _counter_mod._LAST_ACCESS_TIME_COUNTER_FLOAT


def time_of(c: float) -> float:
    """Tick clock at which the counter passed c (the last tick whose counter is <= c)."""
    i = bisect.bisect_right(STATE.tick_c, c) - 1
    if i >= 0:
        return STATE.tick_t[i]
    return STATE.tick_t[0] if STATE.tick_t else STATE.clock


def on_tick_clock(t: float) -> None:
    STATE.clock = t
    c = counter()
    if not STATE.tick_c or c != STATE.tick_c[-1]:
        STATE.tick_c.append(c)
        STATE.tick_t.append(t)


def valid_next_use(node):
    """The node's predicted next use, or None when it has none, was touched since, or is past due."""
    h = STATE.next_use.get(node.id)
    if h is None:
        return None
    t, stamp = h
    if node.last_access_time > stamp - 0.5:
        return None                                   # touched after the hint was set
    if t < NEVER and t < STATE.clock - GRACE:
        return None                                   # past due: LRU order again
    return t


def priority(node) -> float:
    if not node.evicted:
        return node.last_access_time                  # GPU: LRU, as SGLang's default
    t = valid_next_use(node)
    if t is None:
        t = 2 * STATE.clock - time_of(node.last_access_time)   # now + (now - last use)
    return -t


class ParkStrategy(EvictionStrategy):
    """Config (--radix-eviction-policy-config): {"port": 8765} serves late hints on that local port
    (TP rank 0); without it curves can only ride on the request (custom_params park_surv)."""

    def __init__(self, port: int | None = None):
        STATE.port = int(port) if port else None

    def get_priority(self, node) -> float:
        try:
            return priority(node)
        except Exception:  # noqa: BLE001  (never break eviction: fall back to LRU)
            return node.last_access_time


def place(path, t_pred: float) -> int:
    """Set the predicted next use of the path's idle nodes (in use ones are skipped, as in the vLLM
    policy); a node another session placed earlier keeps the earlier use."""
    stamp = counter()
    n = 0
    for node in path:
        if node.component_data[FULL].lock_ref > 0:
            continue
        old = valid_next_use(node)
        STATE.next_use[node.id] = (t_pred if old is None else min(old, t_pred), stamp)
        n += 1
    return n


def valid_before_split(child) -> bool:
    h = STATE.next_use.get(child.id)
    return h is not None and child.last_access_time <= h[1] - 0.5


def on_split(new_node, child, was_valid: bool) -> None:
    """A split keeps the suffix's id (child) and gives the prefix a new node. The split bumps
    child.last_access_time without a use, so a valid hint on the suffix is restamped. The prefix
    starts in LRU order: a split comes from another request matching or inserting through it, which
    is a use. It inherits the open idle periods (split by tokens) and a pending disk placement."""
    h = STATE.next_use.get(child.id)
    if h is not None and was_valid:
        STATE.next_use[child.id] = (h[0], counter())
    for d in (STATE.g_idle, STATE.c_idle):
        r = d.get(child.id)
        if r is not None:
            d[new_node.id] = (r[0], float(len(new_node.key)))
            d[child.id] = (r[0], float(len(child.key)))
    if child.id in STATE.disk_ok:
        STATE.disk_ok.add(new_node.id)
