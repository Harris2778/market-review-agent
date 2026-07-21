#!/usr/bin/env python3
"""自我问责·事后打分 CLI。

用法：
    /usr/local/bin/python3 scripts/score_accountability.py [--days 5] [--archive-dir DIR]

流程：扫描存档目录（ARCHIVE_DIR，缺省 ${DATA_DIR:-data}/archive）下 archive_*.jsonl，
对 score 为 null 且 trade_date 距今 >= days 天的记录，取实际行情涨跌幅打分，
写回 score/scored_at/score_note，并打印 hit/miss/neutral 汇总与明细。

注意：本 CLI 是唯一允许触网的路径（lazy import agent.data_fetcher._get_pro
获取 Tushare 连接）。测试不得覆盖本脚本的真实网络分支。
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import scorer  # noqa: E402

MARKET_INDEX_CODE = "000001.SH"  # market_review / agent_query 的事后核对基准：上证指数
FORWARD_TRADING_DAYS = 5         # 事后核对窗口：trade_date 后 5 个交易日


def resolve_sw_code(sector_name):
    """板块名 → 申万一级行业指数代码（801xx0.SI），找不到返回 None。

    映射逻辑与 agent.data_fetcher._get_sector_member_codes 保持一致：
    先精确匹配 SW_INDEX_MAP，再查俗称别名 SW_SECTOR_ALIAS，最后包含匹配兜底。
    """
    from agent.data_fetcher import SW_INDEX_MAP, SW_SECTOR_ALIAS

    name = (sector_name or "").strip()
    if not name:
        return None
    code = SW_INDEX_MAP.get(name)
    if code:
        return code
    alias = SW_SECTOR_ALIAS.get(name)
    if not alias:
        for key in SW_INDEX_MAP:
            if key in name or name in key:
                alias = key
                break
    if alias:
        return SW_INDEX_MAP.get(alias)
    return None


def pct_change_n_trading_days(pro, ts_code, trade_date, n=FORWARD_TRADING_DAYS):
    """取 ts_code 自 trade_date 收盘起、其后第 n 个交易日收盘的区间涨跌幅（%）。

    数据不足 n+1 行（窗口尚未走完或 trade_date 非交易日）返回 None。
    """
    start = trade_date
    end = (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=n * 3)).strftime("%Y%m%d")
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < n + 1:
        return None
    base = float(df.iloc[0]["close"])
    target = float(df.iloc[n]["close"])
    if base == 0:
        return None
    return round((target - base) / base * 100, 4)


def make_pct_fn(pro):
    """构造生产版 pct_fn：按记录 mode 选择核对基准指数。

    - sector_deep_dive：该板块对应的申万一级行业指数；
    - market_review / agent_query：上证指数 000001.SH。
    任何取数失败（含板块无法映射）都返回 None，由打分层跳过不写回。
    """
    def pct_fn(record):
        mode = record.get("mode")
        if mode == "sector_deep_dive" and record.get("sector"):
            ts_code = resolve_sw_code(record["sector"])
            if not ts_code:
                return None
        else:
            ts_code = MARKET_INDEX_CODE
        try:
            return pct_change_n_trading_days(pro, ts_code, record["trade_date"])
        except Exception:
            return None

    return pct_fn


def main(argv=None):
    parser = argparse.ArgumentParser(description="自我问责·事后打分：核对历史分析的方向判断是否兑现")
    parser.add_argument("--days", type=int, default=5,
                        help="只打分 trade_date 距今 >= 该天数的记录（默认 5）")
    parser.add_argument("--archive-dir", default=scorer.default_archive_dir(),
                        help="存档目录（默认取 ARCHIVE_DIR，缺省 ${DATA_DIR:-data}/archive）")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.archive_dir):
        print(f"存档目录不存在：{args.archive_dir}（无待打分记录）")
        return 0

    # lazy import：仅 CLI 路径允许触网（Tushare）。
    from agent.data_fetcher import _get_pro
    pro = _get_pro()
    if pro is None:
        print("Tushare 连接不可用（TUSHARE_TOKEN 缺失或初始化失败），无法取实际行情，终止。")
        return 1

    result = scorer.apply_scores(args.archive_dir, make_pct_fn(pro), days=args.days)

    counts = {"hit": 0, "miss": 0, "neutral": 0}
    for item in result["scored"]:
        counts[item["score"]] += 1

    print(f"打分完成：新打分 {len(result['scored'])} 条，跳过 {len(result['skipped'])} 条，"
          f"回写文件 {result['files_rewritten']} 个")
    print(f"汇总：hit={counts['hit']}  miss={counts['miss']}  neutral={counts['neutral']}")
    for item in result["scored"]:
        print(f"  [{item['score']:>7}] {item['trade_date']} {item['mode']}"
              f" id={item['id']} — {item['note']}")
    for item in result["skipped"]:
        print(f"  [ skipped] {item['trade_date']} {item['mode']} id={item['id']} — {item['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
