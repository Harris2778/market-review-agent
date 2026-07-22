"""tests/test_daily_update_workflow.py 研报库每日增量自动化测试（零网络零真实执行）。

覆盖范围：
1. workflow 文件存在且含关键字段（cron/workflow_dispatch/concurrency/
   cache/upload-artifact/github-script）；有 PyYAML 则解析断言结构，
   不可用退化为文本断言。
2. cron 表达式恰为 "23 13 * * *"（UTC 13:23 = 北京时间 21:23）。
3. 入口脚本存在、可执行位、bash -n 语法通过；文本含
   PYTHON_BIN/REPORTS_DB_PATH/WITH_FULLTEXT 处理与 set -euo pipefail。
4. 脚本用 fake PYTHON_BIN（打印参数的 stub 可执行文件）在 tmp 目录跑通：
   默认天数=1、--days 透传、爬虫调用参数逐字断言、收尾统计输出。
5. WITH_FULLTEXT=1 时：report_fulltext.py 不存在则跳过并提示、退出码 0；
   存在（v2 就位后）则出现 --days 调用且失败不阻断退出码。

stub 替代真实 python，爬虫/全文脚本绝不真实执行，零网络零外部副作用。
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "daily_report_update.sh"
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "report_crawler.yml"
FULLTEXT_SCRIPT = PROJECT_ROOT / "scripts" / "report_fulltext.py"

EXPECTED_SOURCES = "eastmoney,hibor,stockstar,djyanbao"
EXPECTED_CRON = "23 13 * * *"

try:  # 有 PyYAML 走结构断言，不可用退化为文本断言（同一测试内分支）
    import yaml
except Exception:  # pragma: no cover - 环境无 yaml 时的降级路径
    yaml = None


# ── 公共工具 ──


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _workflow_doc() -> dict:
    """解析 workflow YAML（on 在 YAML 1.1 中会被解析为布尔 True 键）。"""
    assert yaml is not None
    doc = yaml.safe_load(_workflow_text())
    assert isinstance(doc, dict)
    return doc


def _on_section(doc: dict) -> dict:
    on = doc.get("on", doc.get(True))
    assert isinstance(on, dict), "workflow 缺少 on 触发器配置"
    return on


def _steps(doc: dict) -> list:
    steps = doc["jobs"]["crawl"]["steps"]
    assert isinstance(steps, list) and steps
    return steps


def _make_fake_python(tmp_path: Path):
    """生成打印参数的 stub 可执行文件，返回 (stub 路径, 调用日志路径)。"""
    log = tmp_path / "fake_python_calls.log"
    stub = tmp_path / "fake_python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{log}"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub, log


def _run_script(tmp_path: Path, stub: Path, args=(), extra_env=None):
    """用 fake PYTHON_BIN 在隔离环境跑入口脚本，返回 CompletedProcess。

    REPORTS_DB_PATH 指向 tmp 下不存在的库文件：脚本走「库不存在」防御分支，
    不会创建任何文件，也不会触达真实 data/reports.db。
    """
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(stub)
    env["REPORTS_DB_PATH"] = str(tmp_path / "reports.db")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, env=env, timeout=60,
    )


# ── workflow 断言 ──


def test_workflow_file_exists_and_key_tokens():
    """workflow 文件存在，文本含全部关键字段。"""
    assert WORKFLOW.is_file(), "缺少 .github/workflows/report_crawler.yml"
    text = _workflow_text()
    for token in (
        "name: report-crawler-daily",
        EXPECTED_CRON,
        "workflow_dispatch",
        "concurrency",
        "actions/setup-python@v5",
        "actions/cache@v4",
        "actions/upload-artifact@v4",
        "actions/github-script@v7",
        "data/reports.db",
        "reports-db",
        "retention-days: 7",
        "daily_report_update.sh",
        "requirements.txt",
        "ubuntu-latest",
    ):
        assert token in text, f"workflow 缺少关键字段：{token}"


def test_workflow_yaml_structure():
    """解析 workflow YAML 断言结构（无 PyYAML 退化为文本断言）。"""
    if yaml is None:  # 降级：关键结构退化为文本断言
        text = _workflow_text()
        assert "schedule:" in text and "cron:" in text
        assert "workflow_dispatch:" in text
        assert "restore-keys:" in text
        assert 'python-version: "3.14"' in text
        assert "if: failure() && github.event_name == 'schedule'" in text
        return

    doc = _workflow_doc()
    assert doc.get("name") == "report-crawler-daily"

    on = _on_section(doc)
    schedule = on.get("schedule")
    assert isinstance(schedule, list) and schedule, "缺少 schedule 触发器"
    assert any(
        isinstance(item, dict) and item.get("cron") == EXPECTED_CRON
        for item in schedule
    )
    assert "workflow_dispatch" in on

    concurrency = doc.get("concurrency")
    assert isinstance(concurrency, dict) and concurrency.get("group")

    job = doc["jobs"]["crawl"]
    assert job.get("runs-on") == "ubuntu-latest"

    steps = _steps(doc)
    uses = [s.get("uses", "") for s in steps if isinstance(s, dict)]
    for action in (
        "actions/setup-python@v5",
        "actions/cache@v4",
        "actions/upload-artifact@v4",
        "actions/github-script@v7",
    ):
        assert action in uses, f"缺少步骤：{action}"

    setup_py = next(s for s in steps if s.get("uses") == "actions/setup-python@v5")
    assert str(setup_py.get("with", {}).get("python-version")) == "3.14"

    cache = next(s for s in steps if s.get("uses") == "actions/cache@v4")
    cache_with = cache.get("with", {})
    assert cache_with.get("path") == "data/reports.db"
    assert "reports-db-" in str(cache_with.get("key"))
    assert "reports-db-" in str(cache_with.get("restore-keys"))

    upload = next(
        s for s in steps if s.get("uses") == "actions/upload-artifact@v4"
    )
    upload_with = upload.get("with", {})
    assert upload_with.get("name") == "reports-db"
    assert upload_with.get("path") == "data/reports.db"
    assert upload_with.get("retention-days") == 7

    issue = next(s for s in steps if s.get("uses") == "actions/github-script@v7")
    cond = str(issue.get("if", ""))
    assert "failure()" in cond and "schedule" in cond
    script = str(issue.get("with", {}).get("script", ""))
    assert "issues.create" in script and "report-crawler" in script

    crawl = next(s for s in steps if "daily_report_update.sh" in str(s.get("run", "")))
    assert "daily_report_update.sh 1" in crawl["run"]


def test_workflow_cron_expression_exact():
    """cron 恰为 "23 13 * * *"（UTC 13:23 = 北京时间 21:23）。"""
    assert f'cron: "{EXPECTED_CRON}"' in _workflow_text()
    if yaml is not None:
        schedule = _on_section(_workflow_doc()).get("schedule", [])
        crons = [item.get("cron") for item in schedule if isinstance(item, dict)]
        assert crons == [EXPECTED_CRON]


# ── 脚本静态断言 ──


def test_script_exists_and_executable():
    """入口脚本存在且有可执行位。"""
    assert SCRIPT.is_file(), "缺少 scripts/daily_report_update.sh"
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "脚本缺用户可执行位（需 chmod +x）"
    assert os.access(SCRIPT, os.X_OK)


def test_script_bash_syntax_ok():
    """bash -n 语法检查通过（不执行脚本内容）。"""
    res = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30
    )
    assert res.returncode == 0, f"bash -n 语法检查失败：{res.stderr}"


def test_script_env_and_contract_tokens():
    """脚本文本含 PYTHON_BIN/REPORTS_DB_PATH/WITH_FULLTEXT 处理与调用契约。"""
    text = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "set -euo pipefail",
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        'REPORTS_DB_PATH:-data/reports.db',
        'WITH_FULLTEXT:-0',
        "--days",
        "--sources",
        "--db-path",
        EXPECTED_SOURCES,
        "report_crawler.py",
        "report_fulltext.py",
    ):
        assert token in text, f"脚本缺少关键处理：{token}"


# ── 脚本行为断言（fake PYTHON_BIN，零真实执行）──


def test_script_fake_python_default_days(tmp_path):
    """默认天数=1：stub 跑通且爬虫调用参数逐字正确。"""
    stub, log = _make_fake_python(tmp_path)
    res = _run_script(tmp_path, stub)
    assert res.returncode == 0, f"脚本执行失败：{res.stderr}"
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls, "fake PYTHON_BIN 未被调用"
    expected = (
        f"scripts/report_crawler.py --days 1 --sources {EXPECTED_SOURCES} "
        f"--db-path {tmp_path / 'reports.db'}"
    )
    assert calls[0] == expected, f"爬虫调用参数不符：{calls[0]!r}"


def test_script_fake_python_days_passthrough(tmp_path):
    """第一个参数为天数：--days 7 透传给爬虫。"""
    stub, log = _make_fake_python(tmp_path)
    res = _run_script(tmp_path, stub, args=["7"])
    assert res.returncode == 0, f"脚本执行失败：{res.stderr}"
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls and calls[0].startswith(
        f"scripts/report_crawler.py --days 7 --sources {EXPECTED_SOURCES} "
    ), f"--days 透传失败：{calls}"


def test_script_prints_db_stats(tmp_path):
    """收尾打印总篇数与当日新增统计（库不存在走防御分支，退出码 0）。"""
    stub, _ = _make_fake_python(tmp_path)
    res = _run_script(tmp_path, stub)
    assert res.returncode == 0, f"脚本执行失败：{res.stderr}"
    assert "总篇数" in res.stdout and "当日新增" in res.stdout


def test_script_fulltext_handling(tmp_path):
    """WITH_FULLTEXT=1：全文脚本不存在则跳过并提示；存在则调用且不阻断退出码。"""
    stub, log = _make_fake_python(tmp_path)
    res = _run_script(tmp_path, stub, extra_env={"WITH_FULLTEXT": "1"})
    assert res.returncode == 0, f"WITH_FULLTEXT 步骤阻断了退出码：{res.stderr}"
    if FULLTEXT_SCRIPT.exists():
        # v2 全文层已就位：应出现 report_fulltext.py --days 调用
        calls = log.read_text(encoding="utf-8").splitlines()
        assert any(
            "report_fulltext.py" in line and "--days 1" in line for line in calls
        ), f"WITH_FULLTEXT=1 但未调用全文层：{calls}"
    else:
        assert "report_fulltext.py" in res.stdout and "跳过" in res.stdout
