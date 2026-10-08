"""Example policy: write a session to disk only after a human turn (people come back slowly, tools fast).

Run: returnbench eval --policy h2d=examples.human_to_disk:HumanToDisk --rules wt cost
"""


class HumanToDisk:
    def __init__(self, W=None):
        pass

    def decide(self, v):
        human = v["turn"]["kind"] == "human"
        # eta None: the CPU tier keeps its LRU order; disk True: write the session's KV to disk now
        return dict(eta=None, disk=human, pre=None, warm_eta=None)
