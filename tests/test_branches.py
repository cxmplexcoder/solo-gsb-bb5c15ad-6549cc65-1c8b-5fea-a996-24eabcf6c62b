"""app.branches 的单元测试（仅依赖标准库，可直接 `python -m unittest discover`）。"""

from __future__ import annotations

import random
import unittest
from copy import deepcopy
from itertools import product

from app.branches import select_branch
from app.compiler import START, CompileError, compile_commands


def cmd(cid, channel, anchor=START, offset=0, duration=1000):
    return {
        "id": cid,
        "channel": channel,
        "anchor": anchor,
        "offset_ms": offset,
        "duration_ms": duration,
    }


def sw(name, current):
    return {"name": name, "current": current}


def cond(cid, **requires):
    return {"id": cid, "requires": requires}


def ids_of(result):
    return [entry["id"] for entry in result["timeline"]]


class BasicSelectionTests(unittest.TestCase):
    def setUp(self):
        # open: main [0,600)；main_a/main_b: main 锚定 open [600,1500)（互相冲突）；
        # tail: sub 锚定 main_a [1500,1800)
        self.commands = [
            cmd("open", "main", START, 0, 600),
            cmd("main_a", "main", "open", 0, 900),
            cmd("main_b", "main", "open", 0, 900),
            cmd("tail", "sub", "main_a", 0, 300),
        ]
        self.switches = [sw("use_a", True), sw("use_b", False)]
        self.conditions = [
            cond("main_a", use_a=True),
            cond("main_b", use_b=True),
            cond("tail", use_a=True),
        ]

    def test_current_combination_safe_keeps_zero_flips(self):
        # 当前 (use_a=T, use_b=F) 启用 {open, main_a, tail}，无冲突、无悬空锚点
        result = select_branch(self.commands, self.switches, self.conditions)
        self.assertEqual(
            result["selected"],
            [{"name": "use_a", "value": True}, {"name": "use_b", "value": False}],
        )
        self.assertEqual(result["flipped"], [])
        self.assertEqual(ids_of(result), ["open", "main_a", "tail"])
        self.assertEqual(
            result["timeline"],
            [
                entry
                for entry in compile_commands(self.commands)["timeline"]
                if entry["id"] != "main_b"
            ],
        )

    def test_conflict_pair_excludes_combination(self):
        # 当前 (use_a=T, use_b=T) 因 main_a/main_b 冲突无效；
        # 1 翻转邻居 (T,F) 又使锚定 main_b 的 promo 悬空；唯一 1 翻转安全组合为 (F,T)
        commands = self.commands + [cmd("promo", "sub", "main_b", 0, 200)]
        conditions = self.conditions + [cond("promo", use_a=True)]
        result = select_branch(
            commands, [sw("use_a", True), sw("use_b", True)], conditions
        )
        self.assertEqual(
            result["selected"],
            [{"name": "use_a", "value": False}, {"name": "use_b", "value": True}],
        )
        self.assertEqual(result["flipped"], ["use_a"])
        self.assertEqual(ids_of(result), ["open", "main_b"])

    def test_tie_break_prefers_false_then_true_in_input_order(self):
        # base: main [0,500)；x/y: main 锚定 base [500,900)（互相冲突）
        # 当前 (s1=T, s2=T) 冲突无效；1 次翻转的安全组合 (F,T) 与 (T,F) 并列，
        # 按输入顺序 false < true 取 (F,T)
        commands = [
            cmd("base", "main", START, 0, 500),
            cmd("x", "main", "base", 0, 400),
            cmd("y", "main", "base", 0, 400),
        ]
        conditions = [cond("x", s1=True), cond("y", s2=True)]
        result = select_branch(
            commands, [sw("s1", True), sw("s2", True)], conditions
        )
        self.assertEqual(
            result["selected"],
            [{"name": "s1", "value": False}, {"name": "s2", "value": True}],
        )
        self.assertEqual(result["flipped"], ["s1"])
        self.assertEqual(ids_of(result), ["base", "y"])

    def test_dangling_anchor_invalidates_combination(self):
        # promo 锚定 main_b 却跟随 use_a：(use_a=T, use_b=F) 时 promo 启用而
        # main_b 未启用，悬空锚点使该组合无效；最优退到 (F,F)，只留 open
        commands = self.commands + [cmd("promo", "sub", "main_b", 0, 200)]
        conditions = self.conditions + [cond("promo", use_a=True)]
        result = select_branch(commands, self.switches, conditions)
        self.assertEqual(
            result["selected"],
            [{"name": "use_a", "value": False}, {"name": "use_b", "value": False}],
        )
        self.assertEqual(result["flipped"], ["use_a"])
        self.assertEqual(ids_of(result), ["open"])

    def test_flipped_names_follow_switch_input_order(self):
        # c 无条件锚定 a：a 必须启用（否则 c 悬空），故 s1 必须为 true；
        # b 与 a 冲突且要求 s3=false，故 s3 必须为 true；s2 保持当前值
        commands = [
            cmd("a", "main", START, 0, 100),
            cmd("b", "main", START, 0, 100),
            cmd("c", "sub", "a", 0, 50),
        ]
        switches = [sw("s1", False), sw("s2", True), sw("s3", False)]
        conditions = [cond("a", s1=True), cond("b", s3=False)]
        result = select_branch(commands, switches, conditions)
        self.assertEqual(
            result["selected"],
            [
                {"name": "s1", "value": True},
                {"name": "s2", "value": True},
                {"name": "s3", "value": True},
            ],
        )
        self.assertEqual(result["flipped"], ["s1", "s3"])
        self.assertEqual(ids_of(result), ["a", "c"])

    def test_conditionless_commands_always_enabled(self):
        # 没有任何条件：全部口令始终启用，当前组合即最优（0 翻转）
        commands = [cmd("a", "main", START, 0, 100), cmd("b", "sub", START, 0, 100)]
        result = select_branch(commands, [sw("s1", True), sw("s2", False)], [])
        self.assertEqual(
            result["selected"],
            [{"name": "s1", "value": True}, {"name": "s2", "value": False}],
        )
        self.assertEqual(result["flipped"], [])
        self.assertEqual(ids_of(result), ["a", "b"])

    def test_empty_requires_means_always_enabled(self):
        # requires 为空映射等价于未设置条件
        result = select_branch(
            [cmd("a", "main", START, 0, 100)],
            [sw("s", False)],
            [{"id": "a", "requires": {}}],
        )
        self.assertEqual(result["flipped"], [])
        self.assertEqual(ids_of(result), ["a"])

    def test_zero_switches_single_combination(self):
        commands = [cmd("a", "main", START, 0, 100), cmd("b", "sub", "a", 0, 50)]
        result = select_branch(commands, [], [])
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["flipped"], [])
        self.assertEqual(ids_of(result), ["a", "b"])

    def test_zero_switches_with_conflict_has_no_safe_combination(self):
        commands = [cmd("a", "main"), cmd("b", "main")]
        with self.assertRaises(CompileError) as ctx:
            select_branch(commands, [], [])
        self.assertEqual(ctx.exception.code, "NO_FEASIBLE_BRANCH")
        self.assertEqual(
            ctx.exception.details, {"switch_count": 0, "combination_count": 1}
        )


class SafetyAndFeasibilityTests(unittest.TestCase):
    def test_no_safe_combination_raises_structured_error(self):
        # c 无条件且与 a 冲突；b 无条件锚定 a：
        # p=F 时 b 的锚点 a 未启用（悬空），p=T 时 a 与 c 冲突 -> 无安全组合
        commands = [
            cmd("c", "main", START, 0, 1000),
            cmd("a", "main", START, 0, 1000),
            cmd("b", "sub", "a", 0, 500),
        ]
        with self.assertRaises(CompileError) as ctx:
            select_branch(commands, [sw("p", False)], [cond("a", p=True)])
        self.assertEqual(ctx.exception.code, "NO_FEASIBLE_BRANCH")
        self.assertEqual(
            ctx.exception.details, {"switch_count": 1, "combination_count": 2}
        )

    def test_touching_intervals_are_not_conflicts(self):
        # 首尾相接不算冲突：a [0,100) 与 b [100,200) 可同时启用
        commands = [
            cmd("a", "main", START, 0, 100),
            cmd("b", "main", "a", 0, 100),
        ]
        conditions = [cond("a", s=True), cond("b", s=True)]
        result = select_branch(commands, [sw("s", True)], conditions)
        self.assertEqual(result["flipped"], [])
        self.assertEqual(ids_of(result), ["a", "b"])

    def test_cross_channel_overlap_is_not_conflict(self):
        commands = [cmd("a", "main"), cmd("b", "sub")]
        conditions = [cond("a", s=True), cond("b", s=True)]
        result = select_branch(commands, [sw("s", True)], conditions)
        self.assertEqual(ids_of(result), ["a", "b"])

    def test_compile_error_of_full_set_propagates(self):
        # 全量口令编排非法时直接沿用 /compile 错误码
        with self.assertRaises(CompileError) as ctx:
            select_branch([cmd("a", "main", "ghost")], [], [])
        self.assertEqual(ctx.exception.code, "MISSING_ANCHOR")

        with self.assertRaises(CompileError) as ctx:
            select_branch([cmd("a", "main", START, -5, 10)], [], [])
        self.assertEqual(ctx.exception.code, "NEGATIVE_START")

        with self.assertRaises(CompileError) as ctx:
            select_branch([cmd("a", "main"), cmd("a", "sub")], [], [])
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.commands = [cmd("a", "main", START, 0, 100)]

    def test_empty_switch_name(self):
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, [sw("", False)], [])
        self.assertEqual(ctx.exception.code, "INVALID_SWITCH")

    def test_duplicate_switch_name(self):
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, [sw("s", True), sw("s", False)], [])
        self.assertEqual(ctx.exception.code, "DUPLICATE_SWITCH")
        self.assertEqual(ctx.exception.details["names"], ["s"])

    def test_too_many_switches(self):
        switches = [sw(f"s{i}", False) for i in range(9)]
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, switches, [])
        self.assertEqual(ctx.exception.code, "TOO_MANY_SWITCHES")
        self.assertEqual(ctx.exception.details, {"count": 9, "limit": 8})

    def test_eight_switches_allowed(self):
        switches = [sw(f"s{i}", False) for i in range(8)]
        result = select_branch(self.commands, switches, [])
        self.assertEqual(result["flipped"], [])
        self.assertEqual(len(result["selected"]), 8)

    def test_unknown_switch_in_condition(self):
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, [sw("s", False)], [cond("a", ghost=True)])
        self.assertEqual(ctx.exception.code, "UNKNOWN_SWITCH")
        self.assertEqual(ctx.exception.details["names"], ["ghost"])

    def test_unknown_command_id_in_condition(self):
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, [sw("s", False)], [cond("ghost", s=True)])
        self.assertEqual(ctx.exception.code, "BRANCH_TARGET_NOT_FOUND")
        self.assertEqual(ctx.exception.details["ids"], ["ghost"])

    def test_duplicate_condition_id(self):
        conditions = [cond("a", s=True), cond("a", s=False)]
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, [sw("s", False)], conditions)
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")

    def test_empty_condition_id(self):
        with self.assertRaises(CompileError) as ctx:
            select_branch(self.commands, [sw("s", False)], [cond("", s=True)])
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_input_not_mutated(self):
        # 故意省略 anchor / offset_ms 缺省字段，验证 setdefault 不会回写调用方数据
        commands = [
            {"id": "a", "channel": "main", "duration_ms": 100},
            {"id": "b", "channel": "sub", "anchor": "a", "duration_ms": 50},
        ]
        switches = [sw("s", False)]
        conditions = [{"id": "b", "requires": {"s": True}}]
        snapshot = deepcopy((commands, switches, conditions))
        select_branch(commands, switches, conditions)
        self.assertEqual((commands, switches, conditions), snapshot)


class FuzzTests(unittest.TestCase):
    """与独立暴力枚举对照：翻转数最少 + 字典序裁决 + 时间线一致。"""

    def test_matches_brute_force_reference(self):
        rng = random.Random(20260925)
        for _ in range(400):
            n = rng.randint(0, 5)
            commands = []
            for i in range(n):
                anchor = (
                    START
                    if i == 0 or rng.random() < 0.5
                    else f"c{rng.randrange(i)}"  # 只锚定更早的 id，保证无环
                )
                commands.append(
                    cmd(
                        f"c{i}",
                        rng.choice(["main", "sub"]),
                        anchor,
                        rng.randint(0, 3),
                        rng.randint(1, 4),
                    )
                )
            k = rng.randint(0, 3)
            switches = [sw(f"s{j}", rng.random() < 0.5) for j in range(k)]
            conditions = []
            for item in commands:
                if k and rng.random() < 0.5:
                    requires = {
                        f"s{j}": rng.random() < 0.5
                        for j in range(k)
                        if rng.random() < 0.5
                    }
                    conditions.append({"id": item["id"], "requires": requires})

            expected = _reference(commands, switches, conditions)
            if expected is None:
                with self.assertRaises(CompileError) as ctx:
                    select_branch(commands, switches, conditions)
                self.assertEqual(ctx.exception.code, "NO_FEASIBLE_BRANCH")
                continue

            _, combo = expected
            result = select_branch(commands, switches, conditions)
            names = [spec["name"] for spec in switches]
            current = [spec["current"] for spec in switches]
            self.assertEqual([s["name"] for s in result["selected"]], names)
            self.assertEqual([s["value"] for s in result["selected"]], list(combo))
            self.assertEqual(
                result["flipped"],
                [names[j] for j in range(k) if combo[j] != current[j]],
            )
            enabled = _enabled_commands(commands, switches, conditions, combo)
            self.assertEqual(
                result["timeline"], compile_commands(enabled)["timeline"]
            )


def _enabled_commands(commands, switches, conditions, combo):
    """按条件过滤出组合 ``combo`` 下启用的口令（测试侧的独立实现）。"""
    values = dict(zip([spec["name"] for spec in switches], combo))
    requires_by_id = {c["id"]: c["requires"] for c in conditions}
    return [
        item
        for item in commands
        if all(
            values[name] == want
            for name, want in requires_by_id.get(item["id"], {}).items()
        )
    ]


def _reference(commands, switches, conditions):
    """暴力枚举参考实现：安全 = 启用子集可编排且无冲突对。"""
    names = [spec["name"] for spec in switches]
    current = tuple(spec["current"] for spec in switches)
    best = None
    for combo in product((False, True), repeat=len(names)):
        enabled = _enabled_commands(commands, switches, conditions, combo)
        try:
            compiled = compile_commands(deepcopy(enabled))
        except CompileError:
            continue  # 悬空锚点等编排失败 -> 组合无效
        if compiled["conflicts"]:
            continue
        flips = sum(1 for got, cur in zip(combo, current) if got != cur)
        candidate = (flips, combo)
        if best is None or candidate < best:
            best = candidate
    return best


if __name__ == "__main__":
    unittest.main()
