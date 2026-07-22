"""
第五波『Agent 工程三件套』审计与上下文管理层（纯 stdlib，零网络、零 LLM）。

设计灵感来自开源 Agent 项目 Dexter（MIT）的工程实践，本模块全部代码为
本项目原创撰写，仅借鉴其设计思路：
1. Scratchpad 审计日志：把 Agent 循环中的 query / 工具调用 / 工具结果 /
   思考过程逐行落盘为 JSONL，便于事后审计与复盘。
2. 工具调用软护栏（ToolCallGuard）：对单工具过度调用与重复查询生成
   中文软警告文本——只提示、不阻断，由编排层决定如何把警告注入对话。
3. microcompact 上下文管理：当 tool 消息过多或上下文总字符数超预算时，
   把最旧的 tool 消息 content 替换为占位符，严格保持 assistant 的
   tool_calls 与 tool 消息 tool_call_id 的配对关系不变。

工程约定（与 agent/archive.py 对齐）：
- 全部 fail-safe：Scratchpad 任何 OSError 只记 logging.warning，绝不抛出、
  绝不影响主流程。
- 目录惰性解析：Scratchpad 构造时只确定文件名，目录在每次写入/读取 path
  时动态解析——dir_path 参数 > 环境变量 SCRATCHPAD_DIR >
  ${DATA_DIR:-data}/scratchpad（DATA_DIR 为全项目统一数据根目录约定，
  见 DEPLOY.md），便于测试注入与部署切换。
- 时间与随机性可注入：时间戳经模块级 _now_iso() / 文件名时间戳经
  _now_stamp() 生成，测试可 monkeypatch；session_id 缺省用短 uuid。
- 本模块 import 时不拉任何重依赖、不发起任何网络请求。
"""

import difflib
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 阈值常量 ──
# 工具结果序列化后超过该长度即截断（防单个巨型结果撑爆审计文件）
MAX_RESULT_CHARS = 4000
# query / thinking / llm_summary 等自由文本超过该长度即截断
MAX_TEXT_CHARS = 2000
# microcompact 清理后 tool 消息 content 的统一占位符
CLEARED_PLACEHOLDER = "[历史工具结果已清理]"

# 截断后追加的标记（原长指截断前的完整序列化长度）
_TRUNC_SUFFIX = "…[已截断，原长{orig}字符]"


def _now_iso() -> str:
    """当前本地时间 ISO8601（秒级）。独立成函数便于测试 monkeypatch。"""
    return datetime.now().isoformat(timespec="seconds")


def _now_stamp() -> str:
    """当前本地时间紧凑格式 YYYYMMDD_HHMMSS，用于文件名。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(value: Any, max_chars: int = MAX_RESULT_CHARS) -> str:
    """把任意对象兜底序列化为 JSON 字符串。

    - 不可序列化对象经 default=str 转字符串；极端情况下整个序列化失败
      再兜底 str(value)，本函数保证不抛异常；
    - 序列化结果超过 max_chars 时截断并追加原长标记。
    """
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        try:
            text = str(value)
        except Exception:  # noqa: BLE001 - 兜底中的兜底，绝不抛出
            text = "<无法序列化的对象>"
    if len(text) > max_chars:
        text = text[:max_chars] + _TRUNC_SUFFIX.format(orig=len(text))
    return text


class Scratchpad:
    """Agent 会话审计日志（JSONL，best-effort 落盘）。

    文件名为 {YYYYMMDD_HHMMSS}_{session_id 或短uuid}.jsonl，构造时即确定；
    目录惰性解析（见模块 docstring），首次写入时才创建目录与文件。

    所有 log_* 方法均不抛异常：任何 OSError 吞掉并 logging.warning。
    """

    def __init__(self, dir_path: Optional[str] = None, session_id: Optional[str] = None) -> None:
        self._dir_path = dir_path
        # session_id 缺省取 uuid4 前 8 位，足够区分会话且保持文件名简短
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._filename = f"{_now_stamp()}_{self._session_id}.jsonl"
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        """本次会话标识（显式传入或自动生成的短 uuid）。"""
        return self._session_id

    def _resolve_dir(self) -> str:
        """目录惰性解析：dir_path 参数 > SCRATCHPAD_DIR 环境变量 >
        ${DATA_DIR:-data}/scratchpad。每次调用动态读取环境变量，便于测试注入。"""
        if self._dir_path:
            return self._dir_path
        env_dir = os.environ.get("SCRATCHPAD_DIR")
        if env_dir:
            return env_dir
        return os.path.join(os.environ.get("DATA_DIR", "data"), "scratchpad")

    @property
    def path(self) -> str:
        """预期日志文件完整路径（目录尚未初始化时同样可返回，不触碰文件系统）。"""
        return os.path.join(self._resolve_dir(), self._filename)

    def _append(self, entry: Dict[str, Any]) -> None:
        """追加一行 JSONL；任何 OSError 吞掉并记 warning，绝不影响主流程。"""
        try:
            dir_path = self._resolve_dir()
            os.makedirs(dir_path, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False, default=str)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            logger.warning("Scratchpad 写入失败（已忽略，不影响主流程）：%s", exc)

    def log_init(self, query: str) -> None:
        """记录一次 Agent 会话的初始用户查询。"""
        self._append({
            "ts": _now_iso(),
            "type": "init",
            "session_id": self._session_id,
            "query": _json_safe(query, MAX_TEXT_CHARS),
        })

    def log_tool_call(self, name: str, args: Any) -> None:
        """记录一次工具调用（工具名 + 参数）。"""
        self._append({
            "ts": _now_iso(),
            "type": "tool_call",
            "tool": str(name),
            "args": _json_safe(args),
        })

    def log_tool_result(self, name: str, result: Any, llm_summary: Optional[str] = None) -> None:
        """记录一次工具返回结果；llm_summary 可选，为编排层对结果的摘要。"""
        self._append({
            "ts": _now_iso(),
            "type": "tool_result",
            "tool": str(name),
            "result": _json_safe(result, MAX_RESULT_CHARS),
            "llm_summary": _json_safe(llm_summary, MAX_TEXT_CHARS) if llm_summary is not None else None,
        })

    def log_thinking(self, text: str) -> None:
        """记录一段模型思考/推理文本。"""
        self._append({
            "ts": _now_iso(),
            "type": "thinking",
            "text": _json_safe(text, MAX_TEXT_CHARS),
        })


# ── 模块级惰性单例（供 Stage 2 编排层直接取用）──
_default_scratchpad: Optional[Scratchpad] = None
_default_lock = threading.Lock()


def default_scratchpad() -> Scratchpad:
    """返回进程级默认 Scratchpad（惰性单例，首次调用时创建）。"""
    global _default_scratchpad
    if _default_scratchpad is None:
        with _default_lock:
            if _default_scratchpad is None:
                _default_scratchpad = Scratchpad()
    return _default_scratchpad


class ToolCallGuard:
    """工具调用软护栏：只生成中文软警告文本，绝不阻断调用。

    两类警告：
    1. 计数警告——同一工具已调用次数达到上限后，下一次调用（含本次超过
       max_calls_per_tool 次）触发警告，文本含已调用次数与三条出路
       （换工具 / 换参数 / 承认数据缺口并收尾作答）；
    2. 重复查询警告——本次 args 规范化 JSON（sorted keys）与同一工具的
       任一历史调用经 difflib.SequenceMatcher 计算相似度 ≥ 阈值时触发。

    典型用法：编排层先 check(name, args)，把警告文本（若非 None）注入
    对话或日志，然后照常执行工具并 record(name, args)。
    """

    def __init__(self, max_calls_per_tool: int = 3, similarity_threshold: float = 0.7) -> None:
        self.max_calls_per_tool = max_calls_per_tool
        self.similarity_threshold = similarity_threshold
        self._counts: Dict[str, int] = {}
        # 历史调用清单：[(工具名, 规范化 args JSON)]，按调用先后排列
        self._history: List[Tuple[str, str]] = []

    @staticmethod
    def _normalize_args(args: Any) -> str:
        """args 规范化为 sorted-keys JSON 字符串（键序无关，便于相似度比较）。"""
        try:
            return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(args)

    def record(self, name: str, args: Any) -> None:
        """记录一次实际发生的工具调用。"""
        key = str(name)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._history.append((key, self._normalize_args(args)))

    def check(self, name: str, args: Any) -> Optional[str]:
        """检查本次计划中的调用，命中规则时返回中文软警告文本，否则返回 None。"""
        key = str(name)
        called = self._counts.get(key, 0)
        prospective = called + 1  # 含本次在内的调用序号
        if prospective > self.max_calls_per_tool:
            return (
                f"⚠️ 软警告：工具「{key}」已调用 {called} 次，本次为第 {prospective} 次，"
                f"已超过单工具建议上限 {self.max_calls_per_tool} 次。请三选一："
                f"① 换一个工具获取同类信息；"
                f"② 换一组参数（如调整时间窗、更换标的或指标）再查询；"
                f"③ 承认数据缺口，基于已获得的信息收尾作答。"
            )
        norm = self._normalize_args(args)
        for hist_name, hist_args in self._history:
            if hist_name != key:
                continue  # 重复查询仅与同工具历史比较，跨工具相似无意义
            ratio = difflib.SequenceMatcher(None, norm, hist_args).ratio()
            if ratio >= self.similarity_threshold:
                return (
                    f"⚠️ 软警告：工具「{key}」本次查询参数与历史调用高度相似"
                    f"（相似度 {ratio:.0%}，阈值 {self.similarity_threshold:.0%}），"
                    f"疑似重复查询。建议：换一组参数获取增量信息，"
                    f"或承认现有数据已足够并直接收尾作答。"
                )
        return None

    def reset(self) -> None:
        """清空全部计数与历史记录（如新会话开始）。"""
        self._counts.clear()
        self._history.clear()


def _message_chars(message: Dict[str, Any]) -> int:
    """统计单条消息 content 的字符数；None 计 0，非字符串按 str() 长度计。"""
    content = message.get("content")
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    return len(str(content))


def microcompact(
    messages: List[Dict[str, Any]],
    max_tool_msgs: int = 8,
    max_chars: int = 80000,
    keep_recent: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """microcompact 上下文压缩：清理最旧的 tool 消息 content。

    触发条件（满足其一即触发）：
    - role == "tool" 的消息数量 > max_tool_msgs；
    - 全部消息 content 字符总数 > max_chars。

    触发后把除最近 keep_recent 条之外的所有 tool 消息 content 替换为
    CLEARED_PLACEHOLDER；其余字段（含 tool_call_id）原样保留。
    绝不改动 system / user / assistant 消息，绝不改动 assistant 的
    tool_calls 结构，因此 tool_calls 与 tool 消息的配对关系完好。
    注意：若 tool 消息数量 ≤ keep_recent，即使因字符数超限触发，
    受 keep_recent 保护也不会清理任何消息（cleared=0）。

    不原地修改入参：返回由浅拷贝消息组成的新列表，以及统计字典
    {'cleared': 清理条数, 'chars_before': 压缩前总字符数,
     'chars_after': 压缩后总字符数}。
    """
    chars_before = sum(_message_chars(m) for m in messages)
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    triggered = len(tool_indices) > max_tool_msgs or chars_before > max_chars

    # 浅拷贝每条消息：绝不原地修改入参；嵌套的 tool_calls 列表只读不碰
    result = [dict(m) for m in messages]
    cleared = 0
    if triggered:
        protected = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()
        for i in tool_indices:
            if i in protected:
                continue
            if result[i].get("content") != CLEARED_PLACEHOLDER:
                result[i]["content"] = CLEARED_PLACEHOLDER
                cleared += 1

    chars_after = sum(_message_chars(m) for m in result)
    return result, {
        "cleared": cleared,
        "chars_before": chars_before,
        "chars_after": chars_after,
    }
