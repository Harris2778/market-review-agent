"""
第五波 · 定时推送（agent/push.py + main.py 挂载）测试。

覆盖范围：
1. should_fire：工作日越线/恰等到点/未到点/周末/防重复/UTC 自动转上海/
   naive 按上海处理/非法 fire_time 抛 ValueError
2. build_push_payload：文字+图表 URL 组装（mock get_agent 与 charts 模块）、
   charts import 失败降级、图表生成失败降级、快照采集失败降级、
   文字失败不挡图表（双向 fail-safe）
3. send_push：成功 2xx / 超时 / 连接错误 / 非 2xx / 空 URL —— 全部 mock httpx
4. push_tick：不到点不触发、webhook 未配置只 log、配置后发送、防重复
5. main.py 集成：_push_loop 环境变量传递与关闭取消、lifespan 拉起后台任务、
   /charts 静态目录挂载（TestClient 访问临时 svg 返回 200）

所有外部依赖（DeepSeek/Tushare/httpx/charts 模块）全部 mock，绝不发起真实网络请求。
无 pytest-asyncio，异步函数一律用 asyncio.run 驱动（与仓库既有约定一致）。
"""

import asyncio
import os
import sys
import time as time_mod
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# conftest 已注入假 AGENT_API_KEY；setdefault 兜底保证本文件可独立运行
os.environ.setdefault("AGENT_API_KEY", "test-fake-agent-key-for-pytest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
from agent import push  # noqa: E402
from agent.push import (  # noqa: E402
    SHANGHAI_TZ,
    build_push_payload,
    push_tick,
    send_push,
    should_fire,
)

# 2026-07-17 周五 / 07-18 周六 / 07-19 周日 / 07-20 周一（已用解释器核实）
MONDAY = date(2026, 7, 20)
SATURDAY = date(2026, 7, 18)
SUNDAY = date(2026, 7, 19)
FIRE = "15:40"


def _sh(d: date, hh: int, mm: int) -> datetime:
    """构造上海时区时刻。"""
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=SHANGHAI_TZ)


# ═══════════════════════════════════════════
# should_fire
# ═══════════════════════════════════════════

class TestShouldFire:
    def test_weekday_after_fire_time(self):
        assert should_fire(_sh(MONDAY, 15, 41), FIRE) is True

    def test_exactly_at_fire_time_counts_as_crossed(self):
        assert should_fire(_sh(MONDAY, 15, 40), FIRE) is True

    def test_weekday_before_fire_time(self):
        assert should_fire(_sh(MONDAY, 15, 39), FIRE) is False
        assert should_fire(_sh(MONDAY, 9, 30), FIRE) is False

    def test_saturday_never_fires(self):
        assert should_fire(_sh(SATURDAY, 16, 0), FIRE) is False

    def test_sunday_never_fires(self):
        assert should_fire(_sh(SUNDAY, 16, 0), FIRE) is False

    def test_dedup_same_day_already_fired(self):
        assert should_fire(_sh(MONDAY, 15, 41), FIRE, last_fired_date=MONDAY) is False
        assert should_fire(_sh(MONDAY, 18, 0), FIRE, last_fired_date=MONDAY) is False

    def test_fires_again_next_day(self):
        yesterday = MONDAY - timedelta(days=1)
        assert should_fire(_sh(MONDAY, 15, 41), FIRE, last_fired_date=yesterday) is True

    def test_utc_input_converted_to_shanghai(self):
        # UTC 07:41 = 上海 15:41（周一）→ 越线应触发
        utc_dt = datetime(2026, 7, 20, 7, 41, tzinfo=timezone.utc)
        assert should_fire(utc_dt, FIRE) is True

    def test_utc_input_date_rollover(self):
        # UTC 周日 16:41 = 上海周一 00:41 → 已是工作日但未到点
        utc_dt = datetime(2026, 7, 19, 16, 41, tzinfo=timezone.utc)
        assert should_fire(utc_dt, FIRE) is False

    def test_naive_datetime_treated_as_shanghai(self):
        naive = datetime(2026, 7, 20, 15, 41)
        assert should_fire(naive, FIRE) is True

    def test_accepts_time_object(self):
        assert should_fire(_sh(MONDAY, 15, 41), time(15, 40)) is True
        assert should_fire(_sh(MONDAY, 15, 39), time(15, 40)) is False

    def test_whitespace_and_single_digit_hour(self):
        assert should_fire(_sh(MONDAY, 9, 6), " 9:05 ") is True

    @pytest.mark.parametrize("bad", ["15:70", "24:00", "abc", "1540", "15", "15:40:00", ""])
    def test_invalid_fire_time_raises(self, bad):
        with pytest.raises(ValueError):
            should_fire(_sh(MONDAY, 16, 0), bad)


# ═══════════════════════════════════════════
# build_push_payload 的 mock 基础设施
# ═══════════════════════════════════════════

FAKE_CONTENT = "今日市场复盘正文"

CHART_FILES = ["market_overview.svg", "fund_flow.svg"]


def _fake_agent(content=FAKE_CONTENT):
    return SimpleNamespace(
        process_message=AsyncMock(return_value={"content": content})
    )


def _install_fake_charts_module(monkeypatch, chart_files, side_effect=None):
    """向 sys.modules 注入假的 agent.charts（真实文件由并行工程师开发）。"""
    mod = ModuleType("agent.charts")
    if side_effect is not None:
        mod.generate_daily_charts = MagicMock(side_effect=side_effect)
    else:
        mod.generate_daily_charts = MagicMock(return_value=chart_files)
    monkeypatch.setitem(sys.modules, "agent.charts", mod)
    import agent as agent_pkg
    monkeypatch.setattr(agent_pkg, "charts", mod, raising=False)
    return mod


@pytest.fixture()
def chart_dir(tmp_path, monkeypatch):
    """隔离的图表目录：生成 2 个假 svg 文件路径（文件本身不落盘，仅路径）。"""
    d = tmp_path / "charts"
    monkeypatch.setenv("CHART_DIR", str(d))
    files = [str(d / "20260720" / name) for name in CHART_FILES]
    return d, files


# ═══════════════════════════════════════════
# build_push_payload
# ═══════════════════════════════════════════

class TestBuildPushPayload:
    def test_text_and_chart_urls_assembled(self, monkeypatch, chart_dir):
        d, files = chart_dir
        _install_fake_charts_module(monkeypatch, files)
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: _fake_agent())
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(date="20260720")),
        )

        payload = asyncio.run(build_push_payload())

        assert payload["text"] == FAKE_CONTENT
        assert payload["charts"] == [
            "/charts/20260720/market_overview.svg",
            "/charts/20260720/fund_flow.svg",
        ]
        assert payload["trade_date"] == "20260720"

    def test_non_stream_review_called(self, monkeypatch, chart_dir):
        """确认走的是非流式复盘（stream=False）。"""
        d, files = chart_dir
        _install_fake_charts_module(monkeypatch, files)
        agent = _fake_agent()
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: agent)
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(date="20260720")),
        )

        asyncio.run(build_push_payload())

        agent.process_message.assert_awaited_once()
        assert agent.process_message.await_args.kwargs.get("stream") is False

    def test_charts_import_failure_degrades_to_text_only(self, monkeypatch):
        # sys.modules 中置 None 是标准的「import 失败」模拟手法
        monkeypatch.setitem(sys.modules, "agent.charts", None)
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: _fake_agent())

        payload = asyncio.run(build_push_payload())

        assert payload["text"] == FAKE_CONTENT
        assert payload["charts"] == []
        # trade_date 走「上海今天+周末回退」兜底，只校验格式
        assert len(payload["trade_date"]) == 8 and payload["trade_date"].isdigit()

    def test_chart_generation_failure_keeps_text(self, monkeypatch, chart_dir):
        d, files = chart_dir
        _install_fake_charts_module(monkeypatch, None, side_effect=RuntimeError("绘图炸了"))
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: _fake_agent())
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(date="20260720")),
        )

        payload = asyncio.run(build_push_payload())

        assert payload["text"] == FAKE_CONTENT
        assert payload["charts"] == []

    def test_snapshot_failure_keeps_text(self, monkeypatch, chart_dir):
        d, files = chart_dir
        _install_fake_charts_module(monkeypatch, files)
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: _fake_agent())
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(side_effect=RuntimeError("采集超时")),
        )

        payload = asyncio.run(build_push_payload())

        assert payload["text"] == FAKE_CONTENT
        assert payload["charts"] == []

    def test_text_failure_does_not_block_charts(self, monkeypatch, chart_dir):
        """文字复盘失败时图表仍正常生成（图表失败不挡文字的反向）。"""
        d, files = chart_dir
        _install_fake_charts_module(monkeypatch, files)
        bad_agent = SimpleNamespace(
            process_message=AsyncMock(side_effect=RuntimeError("DeepSeek 502"))
        )
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: bad_agent)
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(date="20260720")),
        )

        payload = asyncio.run(build_push_payload())

        assert payload["text"] == ""
        assert payload["charts"] == [
            "/charts/20260720/market_overview.svg",
            "/charts/20260720/fund_flow.svg",
        ]
        assert payload["trade_date"] == "20260720"

    def test_dict_return_from_charts_uses_values(self, monkeypatch, chart_dir):
        d, files = chart_dir
        _install_fake_charts_module(
            monkeypatch, {"overview": files[0], "flow": files[1]}
        )
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: _fake_agent())
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace(date="20260720")),
        )

        payload = asyncio.run(build_push_payload())

        assert sorted(payload["charts"]) == [
            "/charts/20260720/fund_flow.svg",
            "/charts/20260720/market_overview.svg",
        ]

    def test_snapshot_without_date_falls_back(self, monkeypatch, chart_dir):
        d, files = chart_dir
        _install_fake_charts_module(monkeypatch, files)
        monkeypatch.setattr("agent.orchestrator.get_agent", lambda: _fake_agent())
        monkeypatch.setattr(
            "agent.orchestrator.collect_market_snapshot",
            AsyncMock(return_value=SimpleNamespace()),  # 无 date 属性
        )

        payload = asyncio.run(build_push_payload())

        assert len(payload["trade_date"]) == 8 and payload["trade_date"].isdigit()


class TestChartPathToUrl:
    def test_path_outside_chart_dir_uses_parent_dir_name(self):
        url = push._chart_path_to_url("/elsewhere/20260720/a.svg", "/charts_root")
        assert url == "/charts/20260720/a.svg"

    def test_unparseable_path_returns_none(self):
        assert push._chart_path_to_url(None, "charts") is None
        assert push._chart_path_to_url("", "charts") is None


# ═══════════════════════════════════════════
# send_push
# ═══════════════════════════════════════════

class _FakeAsyncClient:
    """httpx.AsyncClient 的假替身：记录调用、按剧本返回/抛错。"""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        _FakeAsyncClient.captured["init_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeAsyncClient.captured["url"] = url
        _FakeAsyncClient.captured["json"] = json
        err = _FakeAsyncClient.captured.get("post_error")
        if err is not None:
            raise err
        return SimpleNamespace(status_code=_FakeAsyncClient.captured.get("status", 200))


@pytest.fixture()
def fake_httpx(monkeypatch):
    _FakeAsyncClient.captured = {}
    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


class TestSendPush:
    URL = "https://example.invalid/webhook"
    PAYLOAD = {"text": "t", "charts": ["/charts/20260720/a.svg"], "trade_date": "20260720"}

    def test_success_returns_true_and_posts_json(self, fake_httpx):
        ok = asyncio.run(send_push(self.URL, self.PAYLOAD))

        assert ok is True
        assert fake_httpx.captured["url"] == self.URL
        assert fake_httpx.captured["json"] == self.PAYLOAD
        # 10 秒超时要求
        assert fake_httpx.captured["init_kwargs"].get("timeout") == 10.0

    def test_timeout_returns_false(self, fake_httpx):
        import httpx
        fake_httpx.captured["post_error"] = httpx.TimeoutException("read timeout")

        assert asyncio.run(send_push(self.URL, self.PAYLOAD)) is False

    def test_connect_error_returns_false(self, fake_httpx):
        import httpx
        fake_httpx.captured["post_error"] = httpx.ConnectError("connection refused")

        assert asyncio.run(send_push(self.URL, self.PAYLOAD)) is False

    def test_non_2xx_returns_false(self, fake_httpx):
        fake_httpx.captured["status"] = 500
        assert asyncio.run(send_push(self.URL, self.PAYLOAD)) is False

    def test_empty_url_returns_false_without_http(self, fake_httpx):
        assert asyncio.run(send_push("", self.PAYLOAD)) is False
        assert "url" not in fake_httpx.captured  # 根本没发起请求


# ═══════════════════════════════════════════
# push_tick（定时循环的单轮逻辑）
# ═══════════════════════════════════════════

class TestPushTick:
    def _patch_build_and_send(self, monkeypatch):
        build = AsyncMock(return_value={
            "text": FAKE_CONTENT,
            "charts": ["/charts/20260720/a.svg"],
            "trade_date": "20260720",
        })
        send = AsyncMock(return_value=True)
        monkeypatch.setattr(push, "build_push_payload", build)
        monkeypatch.setattr(push, "send_push", send)
        return build, send

    def test_not_fired_before_time(self, monkeypatch):
        build, send = self._patch_build_and_send(monkeypatch)
        fired, last = asyncio.run(push_tick(
            fire_time=FIRE, webhook_url=None, last_fired_date=None,
            now=_sh(MONDAY, 15, 39),
        ))
        assert fired is False
        assert last is None
        build.assert_not_awaited()
        send.assert_not_awaited()

    def test_fired_without_webhook_only_logs(self, monkeypatch, caplog):
        build, send = self._patch_build_and_send(monkeypatch)
        with caplog.at_level("INFO", logger="agent.push"):
            fired, last = asyncio.run(push_tick(
                fire_time=FIRE, webhook_url=None, last_fired_date=None,
                now=_sh(MONDAY, 15, 41),
            ))
        assert fired is True
        assert last == MONDAY
        build.assert_awaited_once()
        send.assert_not_awaited()
        assert any("PUSH_WEBHOOK_URL 未配置" in r.message for r in caplog.records)

    def test_fired_with_webhook_sends(self, monkeypatch):
        build, send = self._patch_build_and_send(monkeypatch)
        fired, last = asyncio.run(push_tick(
            fire_time=FIRE, webhook_url="https://example.invalid/hook",
            last_fired_date=None, now=_sh(MONDAY, 15, 41),
        ))
        assert fired is True
        assert last == MONDAY
        send.assert_awaited_once()
        args = send.await_args.args
        assert args[0] == "https://example.invalid/hook"
        assert args[1]["trade_date"] == "20260720"

    def test_already_fired_today_skips(self, monkeypatch):
        build, send = self._patch_build_and_send(monkeypatch)
        fired, last = asyncio.run(push_tick(
            fire_time=FIRE, webhook_url="https://example.invalid/hook",
            last_fired_date=MONDAY, now=_sh(MONDAY, 16, 30),
        ))
        assert fired is False
        assert last == MONDAY
        build.assert_not_awaited()


# ═══════════════════════════════════════════
# main.py 集成：后台循环 / lifespan / 静态挂载
# ═══════════════════════════════════════════

class TestMainIntegration:
    def test_push_loop_reads_env_and_cancels_cleanly(self, monkeypatch):
        """循环读取 PUSH_TIME/PUSH_WEBHOOK_URL 并传给 push_tick；cancel 即退出。"""
        fake = SimpleNamespace(
            DEFAULT_FIRE_TIME="15:40",
            push_tick=AsyncMock(return_value=(False, None)),
        )
        monkeypatch.setenv("PUSH_TIME", "16:05")
        monkeypatch.delenv("PUSH_WEBHOOK_URL", raising=False)

        async def runner():
            with patch.object(main, "push_tasks", fake):
                task = asyncio.create_task(main._push_loop())
                await asyncio.sleep(0.05)  # 让循环跑完第一轮
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(runner())

        assert fake.push_tick.await_count >= 1
        kwargs = fake.push_tick.await_args.kwargs
        assert kwargs["fire_time"] == "16:05"
        assert kwargs["webhook_url"] is None  # 未配置 → None（push_tick 只 log 不发送）

    def test_lifespan_starts_and_stops_push_task(self):
        """TestClient 走完整 lifespan：启动时拉起推送任务，关闭时不悬挂。"""
        from fastapi.testclient import TestClient

        fake = SimpleNamespace(
            DEFAULT_FIRE_TIME="15:40",
            push_tick=AsyncMock(return_value=(False, None)),
        )
        with patch.object(main, "push_tasks", fake):
            with TestClient(main.app) as client:
                deadline = time_mod.time() + 3.0
                while fake.push_tick.await_count < 1 and time_mod.time() < deadline:
                    client.get("/")
                    time_mod.sleep(0.02)
        assert fake.push_tick.await_count >= 1

    def test_charts_mount_registered(self):
        names = [getattr(r, "name", None) for r in main.app.routes]
        assert "charts" in names

    def test_static_files_serves_chart_svg(self):
        """/charts 下临时 svg 应返回 200（含日期子目录层级）。"""
        from fastapi.testclient import TestClient

        chart_root = Path(main.CHART_DIR)
        if not chart_root.is_dir():
            pytest.skip(f"图表目录不可用: {chart_root}")

        uniq = uuid.uuid4().hex[:10]
        sub = chart_root / "20990101"
        f = sub / f"test_push_{uniq}.svg"
        svg = "<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'></svg>"
        sub.mkdir(parents=True, exist_ok=True)
        f.write_text(svg, encoding="utf-8")
        try:
            client = TestClient(main.app)
            resp = client.get(f"/charts/20990101/{f.name}")
            assert resp.status_code == 200
            assert "<svg" in resp.text
        finally:
            f.unlink(missing_ok=True)
            try:
                sub.rmdir()  # 仅在空目录时清理，别动并行工程师的文件
            except OSError:
                pass
