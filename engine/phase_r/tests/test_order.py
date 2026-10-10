"""Arm order randomization, engine-slot assignment, A/A arms and the pairing rule (order.py).

Run: python engine/phase_r/tests/test_order.py
"""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import order  # noqa: E402

ARMS = ["C0", "T", "T2", "LM", "CT", "K", "K_km", "K_ewma"]


class ArmOrder(unittest.TestCase):
    def test_is_a_permutation(self):
        o = order.arm_order(ARMS, 20261010, "main-vllm-slow-heavy", 1)
        self.assertEqual(sorted(o), sorted(ARMS))

    def test_deterministic_and_independent_of_list_order(self):
        a = order.arm_order(ARMS, 20261010, "cell", 3)
        b = order.arm_order(list(reversed(ARMS)), 20261010, "cell", 3)
        self.assertEqual(a, b)
        self.assertEqual(a, order.arm_order(ARMS, 20261010, "cell", 3))

    def test_pinned_draw(self):
        # a change to the draw (hash, key layout) changes every run sheet; this pins it
        self.assertEqual(order.arm_order(["C0", "T", "K"], 1, "c", 1), order.arm_order(["K", "T", "C0"], 1, "c", 1))
        self.assertEqual(order.arm_order(["C0", "T", "K"], 1, "c", 1), sorted(
            ["C0", "T", "K"], key=lambda a: order._key(1, "c", 1, 1, "", a)))

    def test_varies_with_seed_repeat_attempt_cell(self):
        base = order.arm_order(ARMS, 1, "cell", 1)
        self.assertNotEqual(base, order.arm_order(ARMS, 2, "cell", 1))
        self.assertNotEqual(base, order.arm_order(ARMS, 1, "cell", 2))
        self.assertNotEqual(base, order.arm_order(ARMS, 1, "cell", 1, attempt=2))
        self.assertNotEqual(base, order.arm_order(ARMS, 1, "other", 1))

    def test_roughly_uniform_positions(self):
        # over 4000 repeats each arm is first about 1/8 of the time (binomial sd about 21 around 500)
        first = {a: 0 for a in ARMS}
        for r in range(4000):
            first[order.arm_order(ARMS, 7, "cell", r)[0]] += 1
        for a, n in first.items():
            self.assertTrue(400 < n < 600, (a, n))

    def test_three_arms_all_orders_appear(self):
        seen = {tuple(order.arm_order(["C0", "T", "K"], 5, "cell", r)) for r in range(200)}
        self.assertEqual(seen, set(itertools.permutations(["C0", "T", "K"])))

    def test_cell_order(self):
        cells = ["a", "b", "c", "d"]
        self.assertEqual(sorted(order.cell_order(cells, 1, "host1")), cells)
        self.assertEqual(order.cell_order(cells, 1, "host1"), order.cell_order(cells, 1, "host1"))


class Slots(unittest.TestCase):
    def test_one_slot_rounds_follow_order(self):
        r = order.plan_rounds(ARMS, 3, "cell", 1)
        self.assertEqual([x[0][1] for x in r], order.arm_order(ARMS, 3, "cell", 1))
        self.assertTrue(all(len(x) == 1 and x[0][0] == 0 for x in r))

    def test_two_slots_cover_every_arm_once_disjoint(self):
        arms = ["C0", "T", "T2", "K", "K_ewma"]
        for rep in range(1, 7):
            r = order.plan_rounds(arms, 9, "glm", rep, n_slots=2)
            got = [a for rnd in r for _, a in rnd if a != order.FILLER]
            self.assertEqual(sorted(got), sorted(arms))
            self.assertEqual(len(r), 3)
            for rnd in r:
                self.assertEqual(len({s for s, _ in rnd}), len(rnd))   # one arm per slot per round

    def test_an_arm_never_runs_alone_on_two_slots(self):
        # 5 arms over 2 slots: the last round's empty slot runs the filler, so every measured arm has a neighbour
        arms = ["C0", "T", "T2", "K", "K_ewma"]
        for rep in range(1, 5):
            r = order.plan_rounds(arms, 9, "glm", rep, n_slots=2)
            self.assertTrue(all(len(rnd) == 2 for rnd in r), r)
            self.assertEqual(sum(a == order.FILLER for rnd in r for _, a in rnd), 1)
            self.assertTrue(all(sum(a != order.FILLER for _, a in rnd) >= 1 for rnd in r))
            self.assertEqual([s for s, _ in r[-1]], [0, 1])
        self.assertEqual(order.plan_rounds(arms[:4], 9, "glm", 1, n_slots=2)[-1][0][1] != order.FILLER, True)
        self.assertFalse(any(a == order.FILLER for rnd in order.plan_rounds(arms[:4], 9, "glm", 1, n_slots=2)
                             for _, a in rnd))                          # an even split needs none
        self.assertEqual([len(x) for x in order.plan_rounds(arms, 9, "glm", 1, n_slots=2, fill=False)], [2, 2, 1])

    def test_each_arm_on_each_half_equally_across_repeats(self):
        arms = ["C0", "T", "T2", "K", "K_ewma"]
        count = {a: [0, 0] for a in arms}
        for rep in range(1, 5):   # an even number of repeats
            for s, ls in order.assign_slots(arms, rep, 2).items():
                for a in ls:
                    count[a][s] += 1
        self.assertTrue(all(c == [2, 2] for c in count.values()), count)

    def test_aa_pair_runs_together_on_two_halves(self):
        aa = [x["label"] for x in order.aa_arms({"label": "T", "code": "T", "env": {}})]
        self.assertEqual(aa, ["T_aa1", "T_aa2"])
        r = order.plan_rounds(aa, 1, "glm-AA", 1, n_slots=2)
        self.assertEqual(len(r), 1)
        self.assertEqual(sorted(a for _, a in r[0]), aa)

    def test_aa_arms_keep_configuration(self):
        a = {"label": "C0", "code": "C0", "env": {"X": "1"}}
        b = order.aa_arms(a)
        self.assertTrue(all(x["code"] == "C0" and x["env"] == {"X": "1"} for x in b))


class Pairing(unittest.TestCase):
    ok = {"valid": True, "reasons": []}

    def test_all_valid_is_done(self):
        o = order.block_outcome(["C0", "T", "K"], {"C0": self.ok, "T": self.ok, "K": self.ok})
        self.assertEqual(o["excluded"], False)
        self.assertEqual(order.next_step(o, 1, 2), "done")

    def test_one_excluded_arm_reruns_the_whole_block(self):
        bad = {"valid": False, "reasons": ["engine_crash: rc=1"]}
        o = order.block_outcome(["C0", "T", "K"], {"C0": self.ok, "T": bad})
        self.assertTrue(o["excluded"])
        self.assertFalse(o["complete"])
        self.assertEqual(order.next_step(o, 1, 2), "rerun")
        # the rerun is a new attempt of every arm, in a fresh order, not of T alone
        self.assertEqual(sorted(order.arm_order(["C0", "T", "K"], 1, "c", 1, attempt=2)), ["C0", "K", "T"])

    def test_excluded_even_when_complete(self):
        bad = {"valid": False, "reasons": ["memory.events oom +1"]}
        o = order.block_outcome(["C0", "K"], {"C0": self.ok, "K": bad})
        self.assertTrue(o["complete"] and o["excluded"])
        self.assertEqual(order.next_step(o, 1, 3), "rerun")

    def test_host_level_failure_moves_the_block(self):
        o = order.block_outcome(["C0", "K"], {"C0": self.ok},
                                failures=[{"label": "K", "check": "fio", "detail": "-15%", "host_level": True}])
        self.assertTrue(o["host_fault"])
        self.assertEqual(order.next_step(o, 1, 2), "host_fault")

    def test_attempts_used_up_escalates(self):
        bad = {"valid": False, "reasons": ["gpu hw_slowdown"]}
        o = order.block_outcome(["C0", "K"], {"C0": self.ok, "K": bad})
        self.assertEqual(order.next_step(o, 2, 2), "escalate")

    def test_usable_is_the_last_clean_complete_attempt(self):
        bad = {"valid": False, "reasons": ["x"]}
        a1 = {"attempt": 1, "outcome": order.block_outcome(["C0", "K"], {"C0": self.ok, "K": bad}),
              "verdicts": {"C0": self.ok, "K": bad}}
        a2 = {"attempt": 2, "outcome": order.block_outcome(["C0", "K"], {"C0": self.ok, "K": self.ok}),
              "verdicts": {"C0": self.ok, "K": self.ok}}
        self.assertEqual(order.usable([a1, a2])["attempt"], 2)
        self.assertIsNone(order.usable([a1]))
        # C0 of attempt 1 was valid, but never pairs with K of attempt 2
        self.assertNotIn(a1, [order.usable([a1, a2])])


if __name__ == "__main__":
    unittest.main(verbosity=2)
