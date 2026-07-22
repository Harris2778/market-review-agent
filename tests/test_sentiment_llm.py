"""agent/sentiment_llm.py DeepSeek 批量情感打分器测试（全 mock 零网络）。

覆盖范围：
1. JSON 提取各形态：纯净数组 / ```json 代码块包裹 / 前后废话 / 截断无尾括号
   （整批失效 → 词典回退）/ 括号内非法 JSON（整批失效 → 词典回退）。
2. 单条容错：编号错位 / 缺条补齐中性 / 多余条目忽略 / label 非法 / score
   非数值 / score 越界截断 / 数组元素非 dict。
3. 批次失败降级：create 抛异常 → 整批 method='fallback'，label/score 由
   词典法 score_news_sentiment 填好（利好→乐观、利空→悲观归并）；成功批
   与失败批混合时 method 各自正确。
4. client 注入与惰性自建：注入 fake client 被使用、timeout=30/model 透传、
   未注入时走 _build_default_client、构建失败（None）→ 全量回退不抛。
5. 限速：批间 0.5s+抖动、首个批次不限速、全局 index 跨批连续。
6. 边界：空输入 / None 输入 / 非 str 元素强转 / 非法 batch_size 兜底。
7. make_llm_scorer：注入契约（sentiment/sentiment_score/hits）、单条走批量
   接口 batch_size=1、与 score_news_sentiment scorer= 集成、异常绝不抛。

所有 LLM 调用由 FakeClient 注入或 monkeypatch _build_default_client，
sleep 由 fake 记录，绝不触达真实网络与真实 DeepSeek client。
"""

import json

import pytest

import agent.sentiment as sentiment
import agent.sentiment_llm as sl


# ── 公共工具 ──

class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResp:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """openai 风格 chat.completions 替身：handler(kw)->FakeResp 或抛异常。"""

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self._handler(kw)


class FakeChat:
    def __init__(self, handler):
        self.completions = FakeCompletions(handler)


class FakeClient:
    def __init__(self, handler):
        self.chat = FakeChat(handler)


def make_client(handler):
    """构造 fake client：handler(kw)->FakeResp 或抛异常。"""
    return FakeClient(handler)


def static_client(content):
    """返回固定模型输出的 fake client。"""
    return make_client(lambda kw: FakeResp(content))


def raising_client(exc=None):
    """create 必抛异常的 fake client。"""
    def handler(kw):
        raise (exc or RuntimeError("LLM down"))
    return make_client(handler)


def llm_json(entries):
    """构造模型标准输出：[{"i":0,"label":"乐观","score":0.8}, ...]。"""
    return json.dumps(entries, ensure_ascii=False)


def make_sleep(record):
    """构造 fake sleep：记录每次休眠秒数。"""
    def _sleep(seconds):
        record.append(seconds)
    return _sleep


@pytest.fixture(autouse=True)
def _no_default_client(monkeypatch):
    """安全网：默认禁止惰性自建真实 client（缺 key 路径除外，单独 patch）。

    未显式注入 client 且未单独 patch 的用例一律得到 None → 走词典回退，
    保证测试零网络。"""
    monkeypatch.setattr(sl, "_build_default_client", lambda: None)
    yield


# ═══════════════════════════════════════════
# 1. JSON 提取各形态
# ═══════════════════════════════════════════

def test_pure_json_array():
    """纯净 JSON 数组直接解析。"""
    client = static_client(llm_json([
        {"i": 0, "label": "乐观", "score": 0.8},
        {"i": 1, "label": "悲观", "score": -0.6},
    ]))
    out = sl.score_texts_batch(["业绩大增看好", "要崩盘了"], client=client)
    assert [r["label"] for r in out] == ["乐观", "悲观"]
    assert out[0]["score"] == 0.8 and out[1]["score"] == -0.6
    assert all(r["method"] == "llm" for r in out)


def test_json_code_fence_wrapped():
    """```json 代码块包裹的输出可容忍。"""
    content = "```json\n" + llm_json([{"i": 0, "label": "中性", "score": 0.0}]) + "\n```"
    out = sl.score_texts_batch(["今天开盘了"], client=static_client(content))
    assert out == [{"index": 0, "label": "中性", "score": 0.0, "method": "llm"}]


def test_json_with_surrounding_prose():
    """前后废话含方括号：贪婪提取把前导 [ 一并包入 → 非法 JSON → 整批回退。"""
    content = ("好的，分析如下 [这是前导括号]\n"
               + llm_json([{"i": 0, "label": "乐观", "score": 0.5}])
               + "\n以上仅供参考 [完]")
    out = sl.score_texts_batch(["看好后市"], client=static_client(content))
    assert out[0]["index"] == 0
    assert out[0]["method"] == "fallback"  # 提取失效按整批失效处理


def test_json_with_prose_no_brackets():
    """前后废话不含方括号时正常提取成功。"""
    content = "分析结果如下：\n" + llm_json(
        [{"i": 0, "label": "悲观", "score": -0.9}]) + "\n请知悉。"
    out = sl.score_texts_batch(["这票要凉"], client=static_client(content))
    assert out[0]["label"] == "悲观" and out[0]["score"] == -0.9
    assert out[0]["method"] == "llm"


def test_truncated_response_batch_fallback():
    """截断输出（无尾括号）：整批视为 LLM 失效 → 词典回退 method='fallback'。"""
    content = '[{"i": 0, "label": "乐观", "score": 0.8}, {"i": 1, "lab'
    out = sl.score_texts_batch(["涨停 利好", "无关内容"], client=static_client(content))
    assert len(out) == 2
    assert all(r["method"] == "fallback" for r in out)
    # 词典法已填好 label/score：「涨停 利好」命中多头词 → 归并乐观
    assert out[0]["label"] == "乐观" and out[0]["score"] > 0


def test_bracketed_invalid_json_fallback():
    """有括号但内容非法 JSON：整批回退词典。"""
    out = sl.score_texts_batch(["退市 爆雷"], client=static_client("[not json at all]"))
    assert out[0]["method"] == "fallback"
    assert out[0]["label"] == "悲观" and out[0]["score"] < 0


def test_empty_content_fallback():
    """空响应内容：整批回退。"""
    out = sl.score_texts_batch(["业绩增长"], client=static_client(""))
    assert out[0]["method"] == "fallback" and out[0]["label"] == "乐观"


def test_response_top_level_not_array_fallback():
    """JSON 对象为 dict 而非数组：整批回退。"""
    out = sl.score_texts_batch(["预增"], client=static_client('{"i":0,"label":"乐观"}'))
    assert out[0]["method"] == "fallback"


# ═══════════════════════════════════════════
# 2. 单条容错：编号错位/缺条/非法条目
# ═══════════════════════════════════════════

def test_missing_entries_filled_neutral():
    """缺条：模型只回了部分编号，缺的按中性 score=0 补齐（method 仍 llm）。"""
    client = static_client(llm_json([{"i": 0, "label": "乐观", "score": 0.7}]))
    out = sl.score_texts_batch(["看好", "文本乙", "文本丙"], client=client)
    assert [r["label"] for r in out] == ["乐观", "中性", "中性"]
    assert out[1]["score"] == 0.0 and out[2]["score"] == 0.0
    assert all(r["method"] == "llm" for r in out)


def test_misaligned_indices_filled_neutral():
    """编号错位：模型返回的编号与请求不符 → 全部按缺条补中性。"""
    client = static_client(llm_json([
        {"i": 5, "label": "乐观", "score": 0.9},
        {"i": 6, "label": "悲观", "score": -0.9},
    ]))
    out = sl.score_texts_batch(["文本甲", "文本乙"], client=client)
    assert all(r["label"] == "中性" and r["score"] == 0.0 for r in out)
    assert all(r["method"] == "llm" for r in out)


def test_partial_misalignment_mixed_fill():
    """部分编号正确 + 部分错位：正确的采用，错位的忽略、缺条补中性。"""
    client = static_client(llm_json([
        {"i": 1, "label": "悲观", "score": -0.4},
        {"i": 9, "label": "乐观", "score": 0.9},
    ]))
    out = sl.score_texts_batch(["甲", "乙", "丙"], client=client)
    assert [r["label"] for r in out] == ["中性", "悲观", "中性"]
    assert out[1]["score"] == -0.4


def test_extra_entries_ignored():
    """多余编号条目忽略，不报错、不影响有效条目。"""
    client = static_client(llm_json([
        {"i": 0, "label": "乐观", "score": 0.3},
        {"i": 99, "label": "悲观", "score": -1.0},
    ]))
    out = sl.score_texts_batch(["还行"], client=client)
    assert out == [{"index": 0, "label": "乐观", "score": 0.3, "method": "llm"}]


def test_invalid_label_entry_neutralized():
    """label 不在三值内 → 该条丢弃 → 缺条补中性。"""
    client = static_client(llm_json([
        {"i": 0, "label": "利好", "score": 0.8},
        {"i": 1, "label": "悲观", "score": -0.2},
    ]))
    out = sl.score_texts_batch(["甲", "乙"], client=client)
    assert out[0]["label"] == "中性" and out[0]["score"] == 0.0
    assert out[1]["label"] == "悲观"


def test_invalid_score_entry_neutralized():
    """score 非数值 → 该条丢弃 → 补中性。"""
    client = static_client(llm_json([{"i": 0, "label": "乐观", "score": "很高"}]))
    out = sl.score_texts_batch(["甲"], client=client)
    assert out == [{"index": 0, "label": "中性", "score": 0.0, "method": "llm"}]


def test_score_clamped_to_range():
    """score 越界截断到 [-1, 1]。"""
    client = static_client(llm_json([
        {"i": 0, "label": "乐观", "score": 5.0},
        {"i": 1, "label": "悲观", "score": -3.7},
    ]))
    out = sl.score_texts_batch(["甲", "乙"], client=client)
    assert out[0]["score"] == 1.0 and out[1]["score"] == -1.0


def test_non_dict_array_elements_skipped():
    """数组内非 dict 元素丢弃 → 对应编号补中性，其余正常。"""
    client = static_client(
        '["garbage", {"i": 1, "label": "乐观", "score": 0.6}, 42]')
    out = sl.score_texts_batch(["甲", "乙"], client=client)
    assert out[0]["label"] == "中性" and out[0]["score"] == 0.0
    assert out[1]["label"] == "乐观" and out[1]["score"] == 0.6


def test_score_string_number_accepted():
    """score 为数字字符串：宽松解析成功。"""
    client = static_client(llm_json([{"i": 0, "label": "乐观", "score": "0.45"}]))
    out = sl.score_texts_batch(["甲"], client=client)
    assert out[0]["score"] == 0.45 and out[0]["method"] == "llm"


# ═══════════════════════════════════════════
# 3. 批次失败 → 词典回退
# ═══════════════════════════════════════════

def test_create_exception_batch_fallback_lexicon():
    """create 抛异常：整批 method='fallback'，label/score 由词典法填好。"""
    out = sl.score_texts_batch(
        ["涨停 业绩增长 超预期", "退市 爆雷 违约", "今天天气不错"],
        client=raising_client())
    assert len(out) == 3
    assert all(r["method"] == "fallback" for r in out)
    assert out[0]["label"] == "乐观" and out[0]["score"] > 0   # 利好→乐观
    assert out[1]["label"] == "悲观" and out[1]["score"] < 0   # 利空→悲观
    assert out[2]["label"] == "中性" and out[2]["score"] == 0.0


def test_timeout_exception_fallback():
    """超时异常同样整批回退，绝不抛。"""
    out = sl.score_texts_batch(["预增"], client=raising_client(TimeoutError("t")))
    assert out[0]["method"] == "fallback" and out[0]["label"] == "乐观"


def test_mixed_batch_success_and_failure():
    """成功批 method='llm'，失败批 method='fallback'，互不污染。"""
    state = {"n": 0}

    def handler(kw):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("第二批挂了")
        return FakeResp(llm_json([{"i": 0, "label": "乐观", "score": 0.5}]))

    out = sl.score_texts_batch(["甲", "涨停"], client=make_client(handler),
                               batch_size=1, sleep=lambda s: None)
    assert out[0]["method"] == "llm" and out[0]["label"] == "乐观"
    assert out[1]["method"] == "fallback" and out[1]["label"] == "乐观"


def test_fallback_label_mapping_dict_to_llm():
    """回退路径标签归并：利好→乐观、利空→悲观、中性→中性。"""
    out = sl.score_texts_batch(["回购 增持", "减持 处罚", "随便一句"],
                               client=raising_client())
    assert [r["label"] for r in out] == ["乐观", "悲观", "中性"]


# ═══════════════════════════════════════════
# 4. client 注入与惰性自建
# ═══════════════════════════════════════════

def test_client_injection_used():
    """注入的 fake client 被实际调用一次（一个批次）。"""
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    sl.score_texts_batch(["甲"], client=client)
    assert len(client.chat.completions.calls) == 1


def test_timeout_kwarg_passed():
    """LLM 调用透传 timeout=30。"""
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    sl.score_texts_batch(["甲"], client=client)
    assert client.chat.completions.calls[0]["timeout"] == 30


def test_model_param_passed_and_default():
    """model 参数透传；缺省为 deepseek-chat。"""
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    sl.score_texts_batch(["甲"], client=client)
    assert client.chat.completions.calls[0]["model"] == "deepseek-chat"
    sl.score_texts_batch(["甲"], client=client, model="deepseek-reasoner")
    assert client.chat.completions.calls[1]["model"] == "deepseek-reasoner"


def test_prompt_contains_definitions_and_numbering():
    """prompt 含三标签定义与编号列表；system/user 双消息。"""
    client = static_client(llm_json([
        {"i": 0, "label": "乐观", "score": 0.1},
        {"i": 1, "label": "中性", "score": 0.0},
    ]))
    sl.score_texts_batch(["文本零零", "文本壹壹"], client=client)
    messages = client.chat.completions.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "乐观" in user and "悲观" in user and "中性" in user
    assert "看多" in user and "嘲讽看空" in user and "无法判断" in user
    assert "[0] 文本零零" in user and "[1] 文本壹壹" in user
    assert "JSON" in user


def test_lazy_client_built_when_not_injected(monkeypatch):
    """未注入 client 时惰性自建（_build_default_client 被调用并使用）。"""
    fake = static_client(llm_json([{"i": 0, "label": "乐观", "score": 0.2}]))
    calls = []

    def build():
        calls.append(1)
        return fake

    monkeypatch.setattr(sl, "_build_default_client", build)
    out = sl.score_texts_batch(["甲"], client=None)
    assert calls == [1]
    assert out[0]["method"] == "llm" and out[0]["label"] == "乐观"


def test_lazy_client_build_failure_full_fallback():
    """惰性自建失败（None）：全量词典回退，绝不抛。"""
    out = sl.score_texts_batch(["涨停", "跌停"], client=None)
    assert [r["method"] for r in out] == ["fallback", "fallback"]
    assert out[0]["label"] == "乐观" and out[1]["label"] == "悲观"


def test_build_default_client_without_api_key(monkeypatch):
    """真实 _build_default_client：缺 DEEPSEEK_API_KEY 返回 None 不抛。"""
    monkeypatch.undo()  # 恢复真实 _build_default_client
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert sl._build_default_client() is None


# ═══════════════════════════════════════════
# 5. 限速与跨批 index
# ═══════════════════════════════════════════

def test_rate_limit_between_batches():
    """批间限速 0.5s+抖动；首个批次不限速。"""
    sleeps = []
    client = static_client(llm_json([
        {"i": 0, "label": "中性", "score": 0.0},
        {"i": 1, "label": "中性", "score": 0.0},
    ]))
    sl.score_texts_batch(["a", "b", "c", "d", "e"], client=client,
                         batch_size=2, sleep=make_sleep(sleeps))
    assert len(client.chat.completions.calls) == 3  # 3 批
    assert len(sleeps) == 2                          # 批间 2 次限速
    for s in sleeps:
        assert 0.5 <= s <= 0.5 + sl.BATCH_JITTER + 1e-9


def test_single_batch_no_sleep():
    """单批次不触发限速。"""
    sleeps = []
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    sl.score_texts_batch(["only"], client=client, sleep=make_sleep(sleeps))
    assert sleeps == []


def test_sleep_exception_does_not_break():
    """注入的 sleep 抛异常不阻断打分流程。"""
    def bad_sleep(s):
        raise RuntimeError("sleep broken")
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    out = sl.score_texts_batch(["a", "b"], client=client, batch_size=1,
                               sleep=bad_sleep)
    assert len(out) == 2


def test_global_indices_across_batches():
    """跨批 index 全局连续，顺序与输入一致。"""
    def handler(kw):
        # 每批只回 i=0（本批第一条），验证全局 index 映射
        return FakeResp(llm_json([{"i": 0, "label": "乐观", "score": 0.1}]))
    client = make_client(handler)
    texts = ["t0", "t1", "t2", "t3"]
    out = sl.score_texts_batch(texts, client=client, batch_size=2,
                               sleep=lambda s: None)
    assert [r["index"] for r in out] == [0, 1, 2, 3]
    # 每批第一条乐观（i=0），每批第二条缺条补中性
    assert [r["label"] for r in out] == ["乐观", "中性", "乐观", "中性"]


# ═══════════════════════════════════════════
# 6. 边界输入
# ═══════════════════════════════════════════

def test_empty_input_returns_empty_list():
    """空输入返回 []，不发起任何调用。"""
    client = static_client("[]")
    assert sl.score_texts_batch([], client=client) == []
    assert client.chat.completions.calls == []


def test_none_input_returns_empty_list():
    """None 输入返回 []，不抛。"""
    assert sl.score_texts_batch(None) == []


def test_non_str_elements_coerced():
    """非 str 元素强转为 str，不抛。"""
    client = static_client(llm_json([
        {"i": 0, "label": "中性", "score": 0.0},
        {"i": 1, "label": "中性", "score": 0.0},
        {"i": 2, "label": "中性", "score": 0.0},
    ]))
    out = sl.score_texts_batch([123, None, {"x": 1}], client=client)
    assert len(out) == 3
    user = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "[0] 123" in user and "[1] None" in user


def test_invalid_batch_size_fallback_default():
    """非法 batch_size（0/负数/非数值）兜底默认 20，不抛。"""
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    out = sl.score_texts_batch(["a"], client=client, batch_size=0)
    assert len(out) == 1
    out = sl.score_texts_batch(["a"], client=client, batch_size=-3)
    assert len(out) == 1
    out = sl.score_texts_batch(["a"], client=client, batch_size="x")
    assert len(out) == 1
    # 默认 20：21 条 → 2 批
    client2 = static_client("not json")
    sl.score_texts_batch(["x"] * 21, client=client2, sleep=lambda s: None)
    assert len(client2.chat.completions.calls) == 2


# ═══════════════════════════════════════════
# 7. make_llm_scorer 契约
# ═══════════════════════════════════════════

def test_make_llm_scorer_contract_keys():
    """返回 dict 严格含 sentiment/sentiment_score/hits 三键。"""
    scorer = sl.make_llm_scorer(
        client=static_client(llm_json([{"i": 0, "label": "乐观", "score": 0.66}])))
    out = scorer({"title": "看好", "content": "业绩大增"})
    assert out == {"sentiment": "乐观", "sentiment_score": 0.66, "hits": []}


def test_make_llm_scorer_single_uses_batch_api():
    """单条也走批量接口：create 被调一次，timeout=30。"""
    client = static_client(llm_json([{"i": 0, "label": "悲观", "score": -0.3}]))
    scorer = sl.make_llm_scorer(client=client)
    scorer({"title": "要跌"})
    assert len(client.chat.completions.calls) == 1
    assert client.chat.completions.calls[0]["timeout"] == 30


def test_make_llm_scorer_text_join_fields():
    """打分文本拼接 title+summary+content（对齐词典法口径）。"""
    client = static_client(llm_json([{"i": 0, "label": "中性", "score": 0.0}]))
    scorer = sl.make_llm_scorer(client=client)
    scorer({"title": "标题", "summary": "摘要", "content": "正文"})
    user = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "[0] 标题 摘要 正文" in user


def test_make_llm_scorer_integrates_with_score_news_sentiment():
    """作为 scorer 注入 sentiment.score_news_sentiment：契约兼容。"""
    client = static_client(llm_json([{"i": 0, "label": "乐观", "score": 0.9}]))
    scorer = sl.make_llm_scorer(client=client)
    scored = sentiment.score_news_sentiment([{"title": "强烈看好"}], scorer=scorer)
    assert len(scored) == 1
    assert scored[0]["sentiment_score"] == 0.9
    # score_news_sentiment 对非词典三值的 label 按分数归一：0.9 → 利好
    assert scored[0]["sentiment"] == "利好"
    assert scored[0]["hits"] == []


def test_make_llm_scorer_fallback_path():
    """LLM 失败时 scorer 走词典回退结果（method=fallback 的 label/score）。"""
    scorer = sl.make_llm_scorer(client=raising_client())
    out = scorer({"title": "涨停 利好"})
    assert out["sentiment"] == "乐观" and out["sentiment_score"] > 0


def test_make_llm_scorer_never_raises():
    """scorer 任何路径不抛：非 dict 入参 / client 异常均降级中性。"""
    scorer = sl.make_llm_scorer(client=raising_client())
    out = scorer(None)
    assert out == {"sentiment": "中性", "sentiment_score": 0.0, "hits": []}
    out = scorer({"title": "跌停"})
    # 词典回退仍能给出悲观（LLM 挂 ≠ 结果中性）
    assert out["sentiment"] == "悲观"


def test_make_llm_scorer_invalid_label_sanitized():
    """批量结果 label 异常时 scorer 兜底中性（双保险）。"""
    scorer = sl.make_llm_scorer(client=static_client("garbage no array"))
    out = scorer({"title": "随便"})
    assert out["sentiment"] in ("乐观", "中性", "悲观")
    assert isinstance(out["sentiment_score"], float)
