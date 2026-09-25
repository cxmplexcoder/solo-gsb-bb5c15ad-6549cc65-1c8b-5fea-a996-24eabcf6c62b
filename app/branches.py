"""导播分支选择核心逻辑（纯标准库实现，不依赖 Web 框架）。

规则概述
--------
* 现有口令沿用 ``POST /compile`` 的口令数组，先用同一个编排器解析并校验每条口令的
  起止毫秒与同通道冲突。
* 请求另携带至多 8 个名称互异、按输入顺序排列的布尔开关（每个开关给出当前取值），
  以及按口令 id 给出的开关取值条件 ``conditions``：每条条件用一个 ``requires``
  映射声明该口令启用时开关必须满足的取值；**未设置条件的口令始终启用**。
* 枚举全部 ``2^k`` 个开关组合（``k <= 8``，至多 256 个）：对每个组合，只保留
  条件全部成立的口令。保留下来的口令（启用集合）必须满足：

  - 每条启用口令的锚点（``START`` 除外）也在启用集合中——引用了未启用锚点的
    组合无效（悬空锚点）；
  - 启用口令之间不存在同通道半开区间重叠冲突（首尾相接不算冲突）。

  同时满足两者的组合称为安全组合。冲突对直接取编排器对全量口令的解析结果：
  半开区间重叠是口令对的二元性质，与其它口令是否启用无关。
* 在全部安全组合中选择相对开关**当前取值翻转开关数最少**者；翻转数并列时，
  取「按开关输入顺序、``false`` 小于 ``true``」的布尔序列字典序最小者。
* 返回选中的开关取值、发生翻转的开关名（按输入顺序）与启用口令的最终时间线；
  不存在任何安全组合时抛出 ``NO_FEASIBLE_BRANCH``（HTTP 422）。
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from .compiler import START, CompileError, compile_commands

# 一次分支选择最多允许的开关个数
MAX_SWITCHES = 8


class BranchError(CompileError):
    """分支选择失败的业务异常；复用 CompileError 的 422 转换通道。"""


def select_branch(
    commands: list[dict[str, Any]],
    switches: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    """枚举开关组合，选择翻转最少且无悬空锚点、无同通道冲突的安全组合。

    不会修改传入的 ``commands`` / ``switches`` / ``conditions``。

    :raises CompileError: 口令编排非法（重复 id、依赖环、负起点等，与 /compile 相同）。
    :raises BranchError: 开关或条件非法（空名、重名、超过 8 个、未知开关名、
        条件引用未知口令 id），或不存在任何安全组合时抛出，
        ``code`` 可用于程序化区分错误类型。
    """
    # 1) 复用编排器解析并校验全量口令（深拷贝，setdefault 等不会回写调用方数据；
    #    同通道冲突对一并求出：冲突是两两性质，任意子集的冲突恰好是两端都在
    #    该子集中的冲突对）
    compiled = compile_commands(deepcopy(commands))
    entries = compiled["timeline"]
    by_id = {entry["id"]: entry for entry in entries}
    conflict_pairs = compiled["conflicts"]

    # 2) 校验开关定义，保留输入顺序与当前取值
    names, current = _validate_switches(switches)
    k = len(names)

    # 3) 校验按口令 id 的条件，得到每条口令的 (require_mask, require_value)：
    #    requires 中的开关（按开关下标置位）必须取 require_value 对应位的值
    require_mask, require_value = _validate_conditions(
        conditions, by_id, names
    )

    # 冲突口令对的位掩码（按口令在 entries 中的下标编号）：预计算每条口令的
    # 冲突邻居集合，使每个组合的安全性检查只需遍历启用口令（位运算）
    index_of = {entry["id"]: i for i, entry in enumerate(entries)}
    conflict_mask = [0] * len(entries)
    for pair in conflict_pairs:
        ia = index_of[pair["a"]]
        ib = index_of[pair["b"]]
        conflict_mask[ia] |= 1 << ib
        conflict_mask[ib] |= 1 << ia
    # 每条口令的锚点下标（START 记为 -1），悬空锚点检查用
    anchor_index = [
        -1 if entry["anchor"] == START else index_of[entry["anchor"]]
        for entry in entries
    ]
    masks_by_id = {
        entry["id"]: (require_mask.get(entry["id"], 0),
                      require_value.get(entry["id"], 0))
        for entry in entries
    }
    requirement_masks = [masks_by_id[entry["id"]] for entry in entries]

    # 4) 枚举 2^k 个开关组合。product((False, True), repeat=k) 让最后一个开关
    #    翻转最快，元组比较时正好实现「按输入顺序、false < true」的字典序，
    #    因此首个达到某翻转数的安全组合即该翻转数下的字典序最小者。
    best: tuple[int, tuple[bool, ...], int] | None = None
    current_tuple = tuple(current)
    for combo in product((False, True), repeat=k):
        combo_bits = sum(1 << j for j, bit in enumerate(combo) if bit)

        enabled_mask = 0
        for i, (mask, value) in enumerate(requirement_masks):
            if (combo_bits & mask) == value:  # 无条件（mask=0）恒成立
                enabled_mask |= 1 << i

        if not _is_safe(enabled_mask, anchor_index, conflict_mask):
            continue

        flips = sum(1 for j in range(k) if combo[j] != current_tuple[j])
        candidate = (flips, combo, enabled_mask)
        if best is None or candidate < best:
            best = candidate

    # 5) 不存在任何安全组合
    if best is None:
        raise BranchError(
            "NO_FEASIBLE_BRANCH",
            "不存在任何安全开关组合：所有组合都存在悬空锚点或同通道冲突",
            {"switch_count": k, "combination_count": 2**k},
        )

    _, chosen, enabled_mask = best

    # 6) 选中组合的启用集合已保证锚点闭包完整且无冲突，复用编排器重新解析其时间线。
    #    口令字段直接取编排器已补全缺省后的规范投影，绝不回写调用方数据。
    canonical_fields = ("id", "channel", "anchor", "offset_ms", "duration_ms")
    selected_commands = [
        {field: entries[i][field] for field in canonical_fields}
        for i in range(len(entries))
        if enabled_mask & (1 << i)
    ]
    timeline = compile_commands(selected_commands)["timeline"]

    return {
        "selected": [
            {"name": names[j], "value": chosen[j]} for j in range(k)
        ],
        "flipped": [
            names[j] for j in range(k) if chosen[j] != current[j]
        ],
        "timeline": timeline,
    }


def _is_safe(
    enabled_mask: int,
    anchor_index: list[int],
    conflict_mask: list[int],
) -> bool:
    """启用集合是否安全：无同通道冲突口令对，且无悬空锚点。

    只遍历启用的口令位：每条启用口令既不能与另一条启用口令冲突，也不能引用
    未启用的锚点。
    """
    rest = enabled_mask
    while rest:
        low = rest & -rest
        i = low.bit_length() - 1
        rest ^= low
        if conflict_mask[i] & enabled_mask:
            return False
        anchor_idx = anchor_index[i]
        if anchor_idx >= 0 and not enabled_mask & (1 << anchor_idx):
            return False
    return True


def _validate_switches(
    switches: list[dict[str, Any]]
) -> tuple[list[str], list[bool]]:
    """校验开关定义，返回按输入顺序的开关名列表与当前取值列表。"""
    empty = [spec["name"] for spec in switches if spec["name"] == ""]
    if empty:
        raise BranchError(
            "INVALID_SWITCH",
            "开关名不能为空",
            {"names": empty},
        )

    names: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in switches:
        name = spec["name"]
        if name in seen:
            if name not in duplicates:
                duplicates.append(name)
            continue
        seen.add(name)
        names.append(name)
    if duplicates:
        raise BranchError(
            "DUPLICATE_SWITCH",
            f"存在重名的开关: {', '.join(sorted(duplicates))}",
            {"names": sorted(duplicates)},
        )

    if len(names) > MAX_SWITCHES:
        raise BranchError(
            "TOO_MANY_SWITCHES",
            f"开关数量 {len(names)} 超过上限 {MAX_SWITCHES}",
            {"count": len(names), "limit": MAX_SWITCHES},
        )

    return names, [spec["current"] for spec in switches]


def _validate_conditions(
    conditions: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    switch_names: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """校验按口令 id 的开关取值条件。

    返回每条口令的 ``(require_mask, require_value)``：``requires`` 中的开关
    （按其在 ``switch_names`` 中的下标置位）必须取 ``require_value`` 对应位的值。

    :raises BranchError: 条件 id 为空、重复、未知，或 requires 引用未知开关。
    """
    empty = [cond["id"] for cond in conditions if cond["id"] == ""]
    if empty:
        raise BranchError(
            "INVALID_ID",
            "条件的口令 id 不能为空",
            {"ids": empty},
        )

    valid_switches = set(switch_names)
    parsed: list[tuple[str, dict[str, bool]]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    unknown_switches: set[str] = set()
    for cond in conditions:
        cid = cond["id"]
        if cid in seen:
            if cid not in duplicates:
                duplicates.append(cid)
            continue
        seen.add(cid)
        requires = cond["requires"]
        unknown_switches.update(
            name for name in requires if name not in valid_switches
        )
        parsed.append((cid, requires))

    # 未知开关优先于未知口令 id 报告（均为 422，顺序保持确定性）
    if unknown_switches:
        names = sorted(unknown_switches)
        raise BranchError(
            "UNKNOWN_SWITCH",
            f"条件引用了未定义的开关: {', '.join(names)}",
            {"names": names},
        )

    if duplicates:
        raise BranchError(
            "DUPLICATE_ID",
            f"同一口令 id 的条件重复声明: {', '.join(sorted(duplicates))}",
            {"duplicates": sorted(duplicates)},
        )

    unknown_ids = sorted(cid for cid, _ in parsed if cid not in by_id)
    if unknown_ids:
        raise BranchError(
            "BRANCH_TARGET_NOT_FOUND",
            f"条件引用了不存在的口令 id: {', '.join(unknown_ids)}",
            {"ids": unknown_ids},
        )

    require_mask: dict[str, int] = {}
    require_value: dict[str, int] = {}
    for cid, requires in parsed:
        mask = 0
        value = 0
        for j, name in enumerate(switch_names):
            if name in requires:
                mask |= 1 << j
                if requires[name]:
                    value |= 1 << j
        require_mask[cid] = mask
        require_value[cid] = value

    return require_mask, require_value
