"""SGLang general plugin (entry point group sglang.srt.plugins, name "park").

SGLang runs register() in the main process before it parses the server arguments and again in every
scheduler process before the tree cache is built. It adds:
  - the "park" eviction policy (mem_cache.utils._EVICTION_POLICY_FACTORIES, the CLI choices)
  - the "park" radix cache backend (mem_cache.registry), a UnifiedRadixCache subclass
  - the scheduler hooks of layby.sglang.hooks (inert unless the park backend is in use)
The heavy modules (the cache, the hooks) load lazily, in the processes that use them.
"""


def _factory(ctx):
    from layby.sglang.cache import park_cache_factory
    return park_cache_factory(ctx)


def _strategy(**config):
    from layby.sglang.policy import ParkStrategy
    return ParkStrategy(**config)


def register():
    from sglang.srt.arg_groups.choices import RADIX_EVICTION_POLICY_CHOICES, add_radix_eviction_policy_choices
    from sglang.srt.mem_cache import utils as mem_utils
    from sglang.srt.mem_cache.registry import get_radix_cache_factory, register_radix_cache_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    mem_utils._EVICTION_POLICY_FACTORIES["park"] = _strategy
    if "park" not in RADIX_EVICTION_POLICY_CHOICES:
        add_radix_eviction_policy_choices(["park"])
    if get_radix_cache_factory("park") is None:
        register_radix_cache_backend("park", _factory)
    from layby.sglang.hooks import HOOKS
    for target, fn, kind in HOOKS:
        HookRegistry.register(target, fn, HookType[kind])
