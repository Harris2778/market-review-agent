"""多轮对话（history 上下文）测试。

覆盖范围（对应"第一波"多轮对话契约）：
1. 行业切换追问：上文是板块深挖，当前消息含新行业名 → 继承 sector_deep_dive
   并切换到新行业（别名映射：半导体 → 电子）。
2. 原板块追问：上文是板块深挖，当前为对比/追问类消息（无新行业名）
   → 继承 sector_deep_dive 且板块保持原板块。
3. 复盘追问：上文是市场复盘，当前为追问 → 继承 market_review。
4. 无历史不继承：同样的追问消息，history=None → 落回原本意图（general_chat）。
5. 历史截断：history 只保留最近 10 轮（20 条），传入 30 条时
   LLM 收到的 messages 中历史部分不得超过 20 条。
6. _chat 路径带历史：general_chat 时 chat.completions.create 收到的
   messages 必须包含 history 内容且顺序正确。
7. main.py 层：POST /v1/chat/completions 传多轮 messages（含 system），
   agent.process_message 必须收到完整 user/assistant 历史且 system 被过滤。

契约（生产代码并行实现中，本文件按此契约断言）：
- MarketReviewAgent.process_message(user_message, stream=False, history=None)，
  history 为 [{"role": "user"|"assistant", "content": str}]
- main.py 把完整 messages（过滤 system、不含当前最后一条 user）作为 history
  传给 process_message。

规则：
- 所有外部依赖全部 mock（DeepSeek 客户端 / 数据采集 / 子路径方法），
  绝不发起真实网络请求。
- 无 pytest-asyncio，异步函数一律用 asyncio.run 驱动。
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# conftest 会设置 AGENT_API_KEY 测试假值；此处 setdefault 仅作兜底，
# 保证本文件在 conftest 缺席时也可独立运行（main.py 启动强制要求该变量）。
os.environ.setdefault("AGENT_API_KEY", "test-fake-agent-key-for-pytest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.orchestrator import MarketReviewAgent  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ── 共用工具 ──

def _make_agent() -> MarketReviewAgent:
    """构造一个 agent，DeepSeek 客户端替换为 mock，防止任何真实 HTTP 调用。"""
    agent = MarketReviewAgent()
    agent.client = MagicMock()
    return agent


def _fake_completion(content: str = "回复正文", finish_reason: str = "stop"):
    """伪造 chat.completions.create 的非流式返回对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )]
    )


def _run(agent, message, history=None):
    """以非流式驱动 process_message（位置传参，兼容参数命名差异）。"""
    return asyncio.run(agent.process_message(message, False, history))


# 食品饮料板块追问场景的标准历史
SECTOR_HISTORY = [
    {"role": "user", "content": "分析一下食品饮料板块"},
    {"role": "assistant", "content": "食品饮料板块今日表现如下：……（分析正文）"},
]

MARKET_REVIEW_HISTORY = [
    {"role": "user", "content": "今日复盘"},
    {"role": "assistant", "content": "今日市场复盘如下：……（复盘正文）"},
]


# ════════════════════════════════════════════════════════════════
# 1. 行业切换追问：继承 sector_deep_dive 并切换到新行业
# ════════════════════════════════════════════════════════════════

class TestSectorSwitchFollowUp:
    """上文板块深挖 + 当前消息含新行业名 → 继承 sector_deep_dive，板块切换为新行业。"""

    def test_switch_to_new_sector(self):
        """『那半导体呢』→ sector_deep_dive，半导体经别名映射为申万一级『电子』。"""
        agent = _make_agent()
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "电子板块分析"}
        )
        # 兜底：若实现误判意图，其他路径也应被拦截以便定位
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "chat"})
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "review"}
        )

        _run(agent, "那半导体呢", history=SECTOR_HISTORY)

        assert agent._sector_deep_dive.await_count == 1, (
            "上文是食品饮料板块深挖，『那半导体呢』应继承 sector_deep_dive，"
            f"实际 _sector_deep_dive 调用 {agent._sector_deep_dive.await_count} 次"
        )
        sector_arg = agent._sector_deep_dive.await_args.args[0]
        assert sector_arg == "电子", (
            f"半导体应经别名映射切换为申万一级『电子』，实际板块 {sector_arg!r}"
        )
        agent._chat.assert_not_awaited()
        agent._market_review.assert_not_awaited()


# ════════════════════════════════════════════════════════════════
# 2. 原板块追问：继承 sector_deep_dive，板块保持原板块
# ════════════════════════════════════════════════════════════════

class TestSameSectorFollowUp:
    """上文板块深挖 + 对比/追问类消息（无新行业名）→ 继承原板块。"""

    def test_follow_up_keeps_original_sector(self):
        """『跟昨天相比资金怎么样』→ sector_deep_dive，板块仍是食品饮料。

        注：该消息命中落地的追问/对比模式清单（_FOLLOWUP_PATTERNS 的
        对比模式与维度追问模式），且自身不含任何行业名，
        因此应继承上文板块而非切换。"""
        agent = _make_agent()
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "资金流分析"}
        )
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "chat"})
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "review"}
        )

        _run(agent, "跟昨天相比资金怎么样", history=SECTOR_HISTORY)

        assert agent._sector_deep_dive.await_count == 1, (
            "上文是食品饮料板块深挖，追问『跟昨天相比资金怎么样』应继承 sector_deep_dive，"
            f"实际 _sector_deep_dive 调用 {agent._sector_deep_dive.await_count} 次"
        )
        sector_arg = agent._sector_deep_dive.await_args.args[0]
        assert sector_arg == "食品饮料", (
            f"追问未提新行业，板块应保持原板块『食品饮料』，实际 {sector_arg!r}"
        )
        agent._chat.assert_not_awaited()
        agent._market_review.assert_not_awaited()


# ════════════════════════════════════════════════════════════════
# 3. 复盘追问：继承 market_review
# ════════════════════════════════════════════════════════════════

class TestMarketReviewFollowUp:
    """上文市场复盘 + 追问 → 继承 market_review。"""

    def test_follow_up_inherits_market_review(self):
        """『为什么今天跌这么多』→ 继承 market_review。

        注：该消息命中落地的归因追问模式（"为什么…跌"），
        detect_intent 单独识别为 general_chat，必须依靠上文继承。"""
        agent = _make_agent()
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "复盘追问回答"}
        )
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "sector"}
        )
        agent._chat = AsyncMock(return_value={"role": "assistant", "content": "chat"})

        _run(agent, "为什么今天跌这么多", history=MARKET_REVIEW_HISTORY)

        assert agent._market_review.await_count == 1, (
            "上文是市场复盘，追问『为什么今天跌这么多』应继承 market_review，"
            f"实际 _market_review 调用 {agent._market_review.await_count} 次"
        )
        agent._sector_deep_dive.assert_not_awaited()
        agent._chat.assert_not_awaited()


# ════════════════════════════════════════════════════════════════
# 4. 无历史不继承：history=None → 落回原本意图
# ════════════════════════════════════════════════════════════════

class TestNoHistoryNoInheritance:
    """同样的追问消息，没有历史时不发生意图继承。"""

    def test_follow_up_without_history_falls_back(self):
        """『资金流出还会持续吗』本身无明确意图 → general_chat（走 _chat）。"""
        agent = _make_agent()
        agent._chat = AsyncMock(
            return_value={"role": "assistant", "content": "通用回答"}
        )
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "sector"}
        )
        agent._market_review = AsyncMock(
            return_value={"role": "assistant", "content": "review"}
        )

        _run(agent, "资金流出还会持续吗", history=None)

        assert agent._chat.await_count == 1, (
            "history=None 时不应继承上文意图，追问消息应落回 general_chat，"
            f"实际 _chat 调用 {agent._chat.await_count} 次"
        )
        agent._sector_deep_dive.assert_not_awaited()
        agent._market_review.assert_not_awaited()

    def test_follow_up_with_empty_history_falls_back(self):
        """history=[]（空列表）同样不继承。"""
        agent = _make_agent()
        agent._chat = AsyncMock(
            return_value={"role": "assistant", "content": "通用回答"}
        )
        agent._sector_deep_dive = AsyncMock(
            return_value={"role": "assistant", "content": "sector"}
        )

        _run(agent, "资金流出还会持续吗", history=[])

        assert agent._chat.await_count == 1, (
            "history 为空列表时不应发生意图继承，应落回 general_chat"
        )
        agent._sector_deep_dive.assert_not_awaited()


# ════════════════════════════════════════════════════════════════
# 5. 历史截断：history 只保留最近 10 轮（20 条）
# ════════════════════════════════════════════════════════════════

class TestHistoryTruncation:
    """传入 30 条 history，LLM 收到的 messages 中历史部分不得超过 20 条。"""

    def test_history_truncated_to_last_10_rounds(self):
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("回答")
        )

        # 15 轮 × 2 条 = 30 条历史
        history = []
        for i in range(15):
            history.append({"role": "user", "content": f"第{i}轮用户问题"})
            history.append({"role": "assistant", "content": f"第{i}轮助手回答"})

        _run(agent, "随便聊聊现在的事", history=history)

        create = agent.client.chat.completions.create
        assert create.await_count == 1
        messages = create.await_args.kwargs["messages"]

        roles = [m["role"] for m in messages]
        contents = [m["content"] for m in messages]

        # system + 历史 + 当前用户消息；历史截断到 20 条 → 总长不超过 22
        assert roles[0] == "system", "messages 首条应为 system 提示词"
        assert roles[-1] == "user", "messages 末条应为当前用户消息"
        history_part = messages[1:-1]
        assert len(history_part) <= 20, (
            f"历史只应保留最近 10 轮（20 条），实际 LLM 收到 {len(history_part)} 条历史"
        )
        # 最老的历史必须被截掉，最近的必须保留
        assert "第0轮用户问题" not in contents, "最老的一轮历史应被截断丢弃"
        assert "第0轮助手回答" not in contents
        assert "第14轮助手回答" in contents, "最近一轮历史必须保留"
        assert contents[-1] == "随便聊聊现在的事", "当前消息必须在末尾"


# ════════════════════════════════════════════════════════════════
# 6. _chat 路径带历史：LLM 收到的 messages 包含 history 内容
# ════════════════════════════════════════════════════════════════

class TestChatPathCarriesHistory:
    """general_chat 时，chat.completions.create 收到的 messages 必须含历史。"""

    def test_llm_messages_include_history(self):
        agent = _make_agent()
        agent.client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("笑话一则")
        )

        history = [
            {"role": "user", "content": "我喜欢喝咖啡"},
            {"role": "assistant", "content": "好的，我记住了你喜欢喝咖啡"},
        ]

        _run(agent, "给我讲个笑话", history=history)

        create = agent.client.chat.completions.create
        assert create.await_count == 1
        messages = create.await_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        contents = [m["content"] for m in messages]

        assert roles[0] == "system"
        # 历史内容必须按序出现在当前消息之前
        assert "我喜欢喝咖啡" in contents, "LLM 收到的 messages 缺少历史 user 消息"
        assert "好的，我记住了你喜欢喝咖啡" in contents, (
            "LLM 收到的 messages 缺少历史 assistant 消息"
        )
        idx_user = contents.index("我喜欢喝咖啡")
        idx_assistant = contents.index("好的，我记住了你喜欢喝咖啡")
        assert idx_user < idx_assistant, "历史消息顺序必须保持（user 在前 assistant 在后）"
        assert contents[-1] == "给我讲个笑话", "当前消息必须在 messages 末尾"
        # 历史角色必须正确
        hist_msgs = messages[1:-1]
        assert {m["role"] for m in hist_msgs} <= {"user", "assistant"}, (
            "历史消息只允许 user/assistant 角色"
        )


# ════════════════════════════════════════════════════════════════
# 7. main.py 层：完整 messages 传递 + system 过滤
# ════════════════════════════════════════════════════════════════

class TestMainPassesHistory:
    """POST /v1/chat/completions 的多轮 messages 必须完整传给 process_message，
    且 system 消息被过滤。"""

    @pytest.fixture()
    def client(self):
        return TestClient(main.app)

    @pytest.fixture()
    def valid_auth(self):
        return {"Authorization": f"Bearer {main.AGENT_API_KEY}"}

    @staticmethod
    def _extract_call(mock_method):
        """从 AsyncMock 的调用参数中稳健地取出 user_message 与 history。"""
        call = mock_method.await_args
        args, kwargs = call.args, call.kwargs
        user_message = kwargs.get("user_message") or kwargs.get("message")
        if user_message is None and args:
            user_message = args[0]
        if "history" in kwargs:
            history = kwargs["history"]
        elif len(args) >= 3:
            history = args[2]
        else:
            history = None
        return user_message, history

    def test_full_history_passed_and_system_filtered(self, client, valid_auth):
        agent = MagicMock()
        agent.process_message = AsyncMock(
            return_value={"content": "电子板块分析结果"}
        )
        agent.cache_warm = True

        request_messages = [
            {"role": "system", "content": "你是一个自定义系统提示"},
            {"role": "user", "content": "分析一下食品饮料板块"},
            {"role": "assistant", "content": "食品饮料板块分析正文"},
            {"role": "user", "content": "那半导体呢"},
        ]

        with patch.object(main, "get_agent", return_value=agent), \
             patch.object(main, "_agent_loaded", True):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "market-review-agent",
                    "messages": request_messages,
                    "stream": False,
                },
                headers=valid_auth,
            )

        assert resp.status_code == 200, f"请求失败: {resp.status_code} {resp.text}"
        assert agent.process_message.await_count == 1, (
            "process_message 应被调用恰好一次"
        )

        user_message, history = self._extract_call(agent.process_message)

        # 当前消息为最后一条 user 消息
        assert user_message == "那半导体呢", (
            f"process_message 应收到最后一条 user 消息，实际 {user_message!r}"
        )
        # 历史必须包含此前的 user/assistant 轮次
        assert history is not None, "process_message 未收到 history 参数"
        assert {"role": "user", "content": "分析一下食品饮料板块"} in history, (
            f"history 缺少第一轮 user 消息，实际 {history!r}"
        )
        assert {"role": "assistant", "content": "食品饮料板块分析正文"} in history, (
            f"history 缺少第一轮 assistant 消息，实际 {history!r}"
        )
        # system 消息必须被过滤
        roles_in_history = [m["role"] for m in history]
        assert "system" not in roles_in_history, (
            f"system 消息必须被过滤，不得进入 history，实际角色 {roles_in_history}"
        )
        assert all(m["role"] in ("user", "assistant") for m in history)
        # 当前消息不应重复出现在 history 中
        assert {"role": "user", "content": "那半导体呢"} not in history, (
            "当前消息不应同时出现在 history 中"
        )

        # 响应结构保持 OpenAI 兼容
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "电子板块分析结果"
