#!/usr/bin/env python3
"""离线 eval 运行器：加载 eval/cases/*.json，逐条跑确定性 rubric，对比 expect。

用法（项目根目录下）：
    /usr/local/bin/python3 eval/run_eval.py                 # 跑全部 case
    /usr/local/bin/python3 eval/run_eval.py --case <id>     # 只跑指定 case
    /usr/local/bin/python3 eval/run_eval.py --cases-dir <目录>
    /usr/local/bin/python3 eval/run_eval.py --strict        # 严格质量模式

退出码：
    0  全部 case 的实际 rubric 结果与 expect 一致（正例全过、埋雷全被抓出）
    1  任一 case 不一致 / case 文件非法；--strict 模式下任一 rubric 实际失败
    2  运行器自身错误（参数错误等，由 argparse 抛出）

默认模式的判定语义是「expect 匹配」：埋雷 case 的 expect 标注了哪个 rubric
应当失败，实际失败即算该 case 通过——因此包含正确标注埋雷的完整评估集
退出码为 0，可直接接入 CI。--strict 模式忽略 expect，任一 rubric 实际
失败即非零，用于「这批输出必须绝对干净」的场景。

LLM judge 钩子：设置环境变量 EVAL_LLM=1 后，逐 case 在确定性 rubric 之外
追加调用 llm_judge()。当前为预留接口（占位实现直接跳过，不产生任何
网络/LLM 调用），接口约定见 llm_judge docstring 与 eval/README.md。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
# 既能 import 同目录 rubric，也能 import 项目根的 agent 包
for _p in (str(EVAL_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rubric  # noqa: E402

CASES_DIR = EVAL_DIR / "cases"

REQUIRED_FIELDS = ("id", "mode", "fixture_context", "output", "expect")
VALID_MODES = ("market_review", "sector_deep_dive", "news_only")


class CaseError(ValueError):
    """case 文件非法（解析失败 / 缺字段 / expect 非法）。"""


# ── case 加载与校验 ──

def validate_case(case: dict) -> None:
    label = case.get("__file__") or case.get("id") or "<unknown>"
    missing = [f for f in REQUIRED_FIELDS if f not in case]
    if missing:
        raise CaseError("%s: 缺少必需字段 %s（要求 %s）"
                        % (label, missing, list(REQUIRED_FIELDS)))
    if not isinstance(case["id"], str) or not case["id"].strip():
        raise CaseError("%s: id 必须是非空字符串" % label)
    if case["mode"] not in VALID_MODES:
        raise CaseError("%s: mode=%r 非法，合法值 %s"
                        % (label, case["mode"], list(VALID_MODES)))
    for field in ("fixture_context", "output"):
        if not isinstance(case[field], str):
            raise CaseError("%s: %s 必须是字符串" % (label, field))
    expect = case["expect"]
    if not isinstance(expect, dict):
        raise CaseError("%s: expect 必须是 dict（rubric 名 -> 期望布尔）" % label)
    unknown = sorted(set(expect) - set(rubric.RUBRICS))
    if unknown:
        raise CaseError("%s: expect 含未知 rubric %s（已注册 %s）"
                        % (label, unknown, sorted(rubric.RUBRICS)))
    missing_r = sorted(set(rubric.RUBRICS) - set(expect))
    if missing_r:
        raise CaseError("%s: expect 未覆盖全部 rubric，缺 %s" % (label, missing_r))
    for name, want in expect.items():
        if not isinstance(want, bool):
            raise CaseError("%s: expect[%s]=%r 必须是布尔值" % (label, name, want))


def load_cases(cases_dir=CASES_DIR) -> list:
    """加载并校验全部 case 文件，按文件名排序返回 list[dict]。"""
    paths = sorted(Path(cases_dir).glob("*.json"))
    if not paths:
        raise CaseError("未找到任何 case 文件：%s" % cases_dir)
    cases = []
    seen_ids = set()
    for path in paths:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CaseError("%s: JSON 解析失败：%s" % (path.name, exc)) from exc
        if not isinstance(case, dict):
            raise CaseError("%s: 顶层必须是 JSON object" % path.name)
        case["__file__"] = path.name
        validate_case(case)
        if case["id"] in seen_ids:
            raise CaseError("%s: case id %r 重复" % (path.name, case["id"]))
        seen_ids.add(case["id"])
        cases.append(case)
    return cases


# ── LLM judge 预留钩子 ──

def llm_judge(case: dict, rubric_results: list):
    """LLM judge 预留接口（默认关闭，EVAL_LLM=1 时才被调用）。

    实现约定：
    - 入参：case（含 id/mode/fixture_context/output/expect）与确定性
      rubric 的结果清单；
    - 返回：{"name": "llm_judge", "passed": bool, "detail": str}，
      结构与确定性 rubric 一致，会被并入该 case 的判定；
    - 返回 None 表示跳过，不影响判定。

    当前为占位实现：不发起任何网络/LLM 调用，直接返回 None。
    后续接入时在此实现（或替换为外部模块），注意评估环境默认零网络，
    启用方需自行保证凭据与网络可用。
    """
    return None


# ── 评估 ──

def evaluate_case(case: dict, strict: bool = False) -> dict:
    """跑单条 case，返回逐 rubric 对照结果与 case 级判定。

    case 级判定：
    - 默认模式：每个 rubric 的实际 passed 与 expect 一致 → case 通过；
    - --strict：全部 rubric 实际 passed 为 True → case 通过（忽略 expect）。
    """
    results = rubric.run_all(case["output"], case["fixture_context"], case["mode"])
    if os.environ.get("EVAL_LLM") == "1":
        extra = llm_judge(case, results)
        if extra is not None:
            results.append(extra)
    expect = case["expect"]
    checks = []
    for r in results:
        want = expect.get(r["name"])          # llm_judge 等额外结果无 expect
        ok = r["passed"] if strict else (want is None or r["passed"] == want)
        checks.append({
            "name": r["name"],
            "passed": r["passed"],
            "expected": want,
            "ok": ok,
            "detail": r["detail"],
        })
    return {
        "id": case["id"],
        "file": case.get("__file__", "?"),
        "mode": case["mode"],
        "description": case.get("description", ""),
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
    }


def evaluate_all(cases: list, strict: bool = False) -> list:
    return [evaluate_case(c, strict=strict) for c in cases]


# ── CLI ──

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="离线 eval 评估集运行器（零网络、零 LLM）",
    )
    parser.add_argument("--case", help="只跑指定 case id")
    parser.add_argument("--cases-dir", default=str(CASES_DIR),
                        help="case 目录（默认 eval/cases）")
    parser.add_argument("--strict", action="store_true",
                        help="严格质量模式：任一 rubric 实际失败即非零（忽略 expect）")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases_dir)
    except CaseError as exc:
        print("[eval] case 加载失败：%s" % exc, file=sys.stderr)
        return 1

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print("[eval] 未找到 case id：%s" % args.case, file=sys.stderr)
            return 1

    if os.environ.get("EVAL_LLM") == "1":
        print("[eval] EVAL_LLM=1：LLM judge 钩子已启用（当前为预留接口，占位跳过）")

    failed = 0
    for res in evaluate_all(cases, strict=args.strict):
        status = "PASS" if res["passed"] else "FAIL"
        line = "[%s] %s (%s)" % (status, res["id"], res["mode"])
        if res["description"]:
            line += " — %s" % res["description"]
        print(line)
        for c in res["checks"]:
            mark = "OK " if c["ok"] else "XX "
            want = "-" if c["expected"] is None else str(c["expected"])
            print("    %s %-18s passed=%-5s expect=%-5s %s"
                  % (mark, c["name"], str(c["passed"]), want, c["detail"]))
        if not res["passed"]:
            failed += 1

    print("\n[eval] 共 %d 个 case：%d 通过，%d 失败（模式：%s）。"
          % (len(cases), len(cases) - failed, failed,
             "strict" if args.strict else "expect 匹配"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
