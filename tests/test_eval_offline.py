"""离线 eval 评估集的 harness 自守测试（tests/test_eval_offline.py）。

守护对象（零网络、零 LLM、不修改任何现有文件）：
1. eval/cases/*.json 全部可解析、字段齐全、expect 覆盖全部 rubric、
   三种 mode 均有覆盖、每类埋雷至少一个 case；
2. 正例 case（expect 全 True）跑全部 rubric 必须通过；
3. 埋雷 case（expect 含 False）的实际结果必须与 expect 完全一致——
   该抓的抓到、不该抓的不误抓；
4. run_eval.main 退出码语义：完整评估集（埋雷标注正确）返回 0；
   被误标为正例的埋雷 case 返回非零；--strict 下埋雷 case 返回非零；
5. rubric.BANNED_WORDS 与 agent/system_prompts.py 禁用词清单不漂移。

加载方式说明：eval/ 不是包（避免与内置 eval 名字纠缠），这里用
importlib 按文件路径加载 rubric.py 与 run_eval.py；两个模块内部会
自行把项目根加入 sys.path 以 import agent.validators。
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "eval"
CASES_DIR = EVAL_DIR / "cases"

REQUIRED_FIELDS = {"id", "mode", "fixture_context", "output", "expect"}
VALID_MODES = {"market_review", "sector_deep_dive", "news_only"}


def _load_module(name: str, path: Path):
    """按文件路径加载模块（eval/ 非包场景下的确定性加载）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, "无法加载 %s" % path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rubric = _load_module("eval_rubric", EVAL_DIR / "rubric.py")
run_eval = _load_module("eval_run_eval", EVAL_DIR / "run_eval.py")


@pytest.fixture(scope="module")
def cases():
    """加载全部 case；load_cases 本身即做字段/expect 校验，非法会直接抛。"""
    return run_eval.load_cases()


def _is_positive(case) -> bool:
    return all(case["expect"].values())


def _mined_expectations(case):
    """该 case 中期望被 rubric 抓出的项：{rubric_name: False}。"""
    return {k: v for k, v in case["expect"].items() if v is False}


# ─────────────────────────────────────────────
# 1. case 文件：可解析、字段齐全、覆盖度
# ─────────────────────────────────────────────

class TestCaseFiles:
    def test_cases_count(self, cases):
        files = list(CASES_DIR.glob("*.json"))
        assert 10 <= len(files) <= 12, "任务要求 10~12 个 case，实际 %d" % len(files)
        assert len(cases) == len(files)

    def test_fields_complete(self, cases):
        ids = set()
        for case in cases:
            assert REQUIRED_FIELDS <= set(case), \
                "%s 缺字段 %s" % (case.get("__file__"), REQUIRED_FIELDS - set(case))
            assert case["id"] not in ids, "case id 重复：%s" % case["id"]
            ids.add(case["id"])
            assert case["mode"] in VALID_MODES
            assert isinstance(case["fixture_context"], str) \
                and case["fixture_context"].strip()
            assert isinstance(case["output"], str) and case["output"].strip()
            # expect 必须覆盖全部 rubric 且值为布尔
            assert set(case["expect"]) == set(rubric.RUBRICS)
            assert all(isinstance(v, bool) for v in case["expect"].values())

    def test_all_modes_covered(self, cases):
        modes = {c["mode"] for c in cases}
        assert VALID_MODES <= modes, "三种 mode 都要有 case，缺 %s" % (VALID_MODES - modes)

    def test_every_mine_type_has_a_case(self, cases):
        """每类雷（每个 rubric）至少一个 expect=False 的 case。"""
        mined = set()
        for case in cases:
            mined.update(_mined_expectations(case))
        required_mines = {
            "number_sourcing",   # 无出处数字
            "banned_words",      # 禁用词
            "markdown_table",    # 表格残留
            "risk_disclaimer",   # 缺风险提示
            "sector_structure",  # 五维结构缺失
        }
        assert required_mines <= mined, "缺埋雷类型：%s" % (required_mines - mined)

    def test_has_both_positive_and_mined(self, cases):
        assert any(_is_positive(c) for c in cases), "至少一个正例"
        assert any(not _is_positive(c) for c in cases), "至少一个埋雷反例"


# ─────────────────────────────────────────────
# 2/3. rubric 实际结果 vs expect
# ─────────────────────────────────────────────

class TestRubricOutcomes:
    def test_positive_cases_pass_every_rubric(self, cases):
        for case in cases:
            if not _is_positive(case):
                continue
            for r in rubric.run_all(case["output"], case["fixture_context"],
                                    case["mode"]):
                assert r["passed"], \
                    "正例 %s 被 %s 误抓：%s" % (case["id"], r["name"], r["detail"])

    def test_mined_cases_caught_exactly_as_expected(self, cases):
        for case in cases:
            if _is_positive(case):
                continue
            results = {r["name"]: r for r in rubric.run_all(
                case["output"], case["fixture_context"], case["mode"])}
            for name, want in case["expect"].items():
                got = results[name]["passed"]
                assert got is want, (
                    "埋雷 case %s 的 rubric %s：期望 passed=%s，实际 %s（%s）"
                    % (case["id"], name, want, got, results[name]["detail"])
                )

    def test_evaluate_case_all_match_expect(self, cases):
        """走 run_eval.evaluate_case 全链路：每个 case 都应与 expect 一致。"""
        for case in cases:
            res = run_eval.evaluate_case(case)
            assert res["passed"], \
                "case %s 与 expect 不一致：%s" % (
                    case["id"],
                    [c for c in res["checks"] if not c["ok"]])


# ─────────────────────────────────────────────
# 4. run_eval.main 退出码语义
# ─────────────────────────────────────────────

class TestRunEvalMain:
    def test_full_suite_exit_zero(self, capsys):
        """完整评估集（埋雷标注正确）：正例过、反例被抓 → 退出码 0。"""
        rc = run_eval.main(["--cases-dir", str(CASES_DIR)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "0 失败" in out

    def test_single_case_filter(self, capsys):
        rc = run_eval.main(["--cases-dir", str(CASES_DIR),
                            "--case", "market_review_basic_ok"])
        capsys.readouterr()
        assert rc == 0

    def test_mislabeled_mined_case_exit_nonzero(self, tmp_path, capsys):
        """埋雷 case 被误标为正例（expect 全 True）→ main 必须返回非零。"""
        src = None
        for p in sorted(CASES_DIR.glob("*.json")):
            case = json.loads(p.read_text(encoding="utf-8"))
            if not all(case["expect"].values()):
                src = (p, case)
                break
        assert src is not None, "评估集里找不到埋雷 case"
        path, case = src
        case["expect"] = {k: True for k in case["expect"]}  # 误标为正例
        (tmp_path / path.name).write_text(
            json.dumps(case, ensure_ascii=False), encoding="utf-8")
        rc = run_eval.main(["--cases-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc != 0, "误标埋雷 case 未被 main 判失败"
        assert "FAIL" in out

    def test_strict_mode_flags_mined_case(self, tmp_path, capsys):
        """--strict 忽略 expect：埋雷 case 任一 rubric 实际失败 → 非零。"""
        src = None
        for p in sorted(CASES_DIR.glob("*.json")):
            case = json.loads(p.read_text(encoding="utf-8"))
            if not all(case["expect"].values()):
                src = p
                break
        assert src is not None
        shutil.copy(src, tmp_path / src.name)
        rc_strict = run_eval.main(["--cases-dir", str(tmp_path), "--strict"])
        rc_default = run_eval.main(["--cases-dir", str(tmp_path)])
        capsys.readouterr()
        assert rc_strict != 0, "--strict 下埋雷 case 应返回非零"
        assert rc_default == 0, "默认模式下标注正确的埋雷 case 应返回 0"

    def test_invalid_case_dir_exit_nonzero(self, tmp_path, capsys):
        """空目录（无 case）→ 非零。"""
        rc = run_eval.main(["--cases-dir", str(tmp_path)])
        capsys.readouterr()
        assert rc != 0


# ─────────────────────────────────────────────
# 5. 禁用词清单与 system_prompts 不漂移
# ─────────────────────────────────────────────

class TestBannedWordsSync:
    def test_banned_words_present_in_system_prompts(self):
        import agent.system_prompts as sp
        blob = "\n".join(
            v for k, v in vars(sp).items()
            if k.isupper() and isinstance(v, str)
        )
        for w in rubric.BANNED_WORDS:
            assert w in blob, \
                "禁用词 %r 未出现在 agent/system_prompts.py，请同步 rubric.BANNED_WORDS" % w

    def test_risk_disclaimer_consistent_with_orchestrator(self):
        """rubric 的风险提示语关键句与 orchestrator 追加的免责声明一致。"""
        src = (PROJECT_ROOT / "agent" / "orchestrator.py").read_text(encoding="utf-8")
        assert rubric.RISK_DISCLAIMER in src

    def test_sector_dimensions_match_prompt_framework(self):
        """五维标记必须与 SECTOR_DEEP_DIVE_PROMPT 的分析框架小节对应。"""
        from agent.system_prompts import SECTOR_DEEP_DIVE_PROMPT
        for d in rubric.SECTOR_DIMENSIONS:
            assert d in SECTOR_DEEP_DIVE_PROMPT, \
                "五维标记 %r 未出现在 SECTOR_DEEP_DIVE_PROMPT" % d
