"""Arm order, engine-slot assignment and the pairing rule (BENCH_PLAN sections 5 and 7).

Order. Arms (and cells) run in a random order per host, drawn from the seed fixed in the plan. The draw sorts the arms
by sha256(seed | cell | repeat | attempt | arm), so it is a uniform permutation that does not depend on the Python
version, on the order the arms are listed in, or on any other cell. A block that reruns (attempt 2) draws a new order.

Slots. A GLM node runs two engines at once, one per NUMA half. Arm i of the cell's list goes to slot
(i + repeat) mod n_slots, so over n_slots consecutive repeats every arm runs on every slot once; within a slot the
order is the seeded draw. Slots run in lockstep rounds: round k starts slot A's k-th and slot B's k-th arm together,
and the between-arm reset (drop caches, wipe, probe) happens with both stopped. When the arms do not divide evenly,
the last round has an empty slot; it runs FILLER (an unscored C0 replay of the same workload, outputs discarded), so
no measured arm runs alone on a node built for two engines.

Pairing rule. A block is all arms of one cell on one host in one repeat. An arm excluded for a failed health check,
a throttle event, an engine crash or a host fault excludes its whole block, which reruns, so every comparison stays
within one host. Results of an excluded block attempt are kept and reported, never used for the endpoints.
A host-level failure (GPU throttle, fio drift, a provider fault) means the block reruns on a replacement host.
"""
import hashlib
from itertools import zip_longest

FILLER = "_filler"     # the unscored neighbour in a round with an empty slot (labels may not start with _)


def _key(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def arm_order(labels, seed, cell, repeat, attempt=1, slot=None):
    tag = "" if slot is None else f"slot{slot}"
    return sorted(labels, key=lambda a: _key(seed, cell, repeat, attempt, tag, a))


def cell_order(cells, seed, host_key):
    """Order of cells on one host (host_key: the host's id or its repeat index, as the plan's run sheet fixes)."""
    return sorted(cells, key=lambda c: _key(seed, "cells", host_key, c))


def assign_slots(labels, repeat, n_slots):
    """slot index -> labels (in list order), arm i to slot (i + repeat) % n_slots."""
    out = {s: [] for s in range(n_slots)}
    for i, a in enumerate(labels):
        out[(i + repeat) % n_slots].append(a)
    return out


def plan_rounds(labels, seed, cell, repeat, attempt=1, n_slots=1, fill=True):
    """List of rounds; each round is a list of (slot, label) run together, in slot order. With fill, an empty slot
    in a round runs FILLER."""
    if n_slots == 1:
        return [[(0, a)] for a in arm_order(labels, seed, cell, repeat, attempt)]
    per = {s: arm_order(ls, seed, cell, repeat, attempt, slot=s) for s, ls in assign_slots(labels, repeat, n_slots).items()}
    rounds = []
    for row in zip_longest(*(per[s] for s in range(n_slots))):
        rounds.append([(s, a if a is not None else FILLER) for s, a in enumerate(row) if a is not None or fill])
    return rounds


def aa_arms(arm):
    """A/A mode: the arm twice, as two labelled arms with the same configuration."""
    return [dict(arm, label=f"{arm['label']}_aa1"), dict(arm, label=f"{arm['label']}_aa2")]


def block_outcome(planned, verdicts, failures=()):
    """planned: labels the block must run; verdicts: label -> verdict dict ({'valid', 'reasons'}) for arms that ran;
    failures: pre-arm health failures [{'label', 'check', 'detail', 'host_level'}].
    Returns {'complete', 'excluded', 'host_fault', 'reasons'}."""
    reasons, host_fault = [], False
    for f in failures:
        reasons.append(f"{f['label']}: health {f['check']}: {f['detail']}")
        host_fault |= bool(f.get("host_level"))
    for lab, v in verdicts.items():
        if not v["valid"]:
            reasons += [f"{lab}: {r}" for r in v["reasons"]]
    missing = [a for a in planned if a not in verdicts]
    complete = not missing
    if missing and not reasons:
        reasons.append(f"not run: {missing}")
    return {"complete": complete, "excluded": bool(reasons), "host_fault": host_fault, "reasons": reasons}


def next_step(outcome, attempt, max_attempts):
    """done | rerun (same host) | host_fault (rerun on a replacement host) | escalate (owner decides)."""
    if not outcome["excluded"] and outcome["complete"]:
        return "done"
    if attempt >= max_attempts:
        return "escalate"
    if outcome["host_fault"]:
        return "host_fault"
    return "rerun"


def usable(attempts):
    """The verdict map of the block attempt the analysis uses: the last one that completed with no exclusion. Every
    other attempt is reported only. None if no attempt qualifies. attempts: [{'attempt', 'outcome', 'verdicts'}]."""
    good = [a for a in attempts if a["outcome"]["complete"] and not a["outcome"]["excluded"]]
    return max(good, key=lambda a: a["attempt"]) if good else None
