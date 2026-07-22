#!/usr/bin/env bash
# scripts/daily_report_update.sh —— 研报库每日增量统一入口（研报库 v3 自动化）。
#
# 职责：
# 1. 调用四源爬虫（东财/慧博/证券之星/洞见）做增量抓取并入 SQLite 库；
# 2. 收尾打印库内总篇数与当日新增统计（created_at 为当日的首写记录数）；
# 3. WITH_FULLTEXT=1 时追加全文层抓取（scripts/report_fulltext.py，v2 全文层
#    由并行工作流实现，此处只对接口不对实现：仅传 --days；失败仅 warning 不
#    阻断退出码；文件不存在则跳过并提示）。
#
# 用法：
#   scripts/daily_report_update.sh [天数]                     # 默认 1=当日增量
#   PYTHON_BIN=/usr/local/bin/python3 scripts/daily_report_update.sh 1
#   REPORTS_DB_PATH=data/reports.db WITH_FULLTEXT=1 scripts/daily_report_update.sh 1
#
# 环境变量：
#   PYTHON_BIN       Python 解释器（默认 python3）
#   REPORTS_DB_PATH  研报库路径（默认 data/reports.db；与 agent.report_library
#                    的惰性解析契约一致：显式设置 > REPORTS_DB_PATH >
#                    ${DATA_DIR:-data}/reports.db）
#   WITH_FULLTEXT    =1 时追加全文层抓取 + 向量索引增量重建（默认关闭）

set -euo pipefail

# cd 到仓库根（本脚本位于 scripts/ 下，任意 cwd 调用均可）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DB_PATH="${REPORTS_DB_PATH:-data/reports.db}"

# 第一个参数为回溯天数（默认 1）；非法输入告警并回退 1
DAYS="${1:-1}"
if ! [[ "${DAYS}" =~ ^[0-9]+$ ]] || [ "${DAYS}" -lt 1 ]; then
  echo "WARNING: 天数参数非法（${DAYS}），回退为 1" >&2
  DAYS=1
fi

SOURCES="eastmoney,hibor,stockstar,djyanbao"

echo "== 研报库每日增量更新 =="
echo "Python: ${PYTHON_BIN}；库: ${DB_PATH}；回溯天数: ${DAYS}；源: ${SOURCES}"

# ── 1. 四源元数据增量抓取（爬虫自身按源 fail-safe，非零退出=硬失败）──
"${PYTHON_BIN}" scripts/report_crawler.py \
  --days "${DAYS}" \
  --sources "${SOURCES}" \
  --db-path "${DB_PATH}"

# ── 2. 收尾统计：库内总篇数 + 当日新增（created_at 日期=当日）──
print_db_stats() {
  local db="$1" total="0" today="0"
  if [ ! -f "${db}" ]; then
    echo "统计：研报库文件不存在（${db}），总篇数 0，当日新增 0"
    return 0
  fi
  if command -v sqlite3 >/dev/null 2>&1; then
    total="$(sqlite3 "${db}" "SELECT COUNT(*) FROM reports;" 2>/dev/null || echo 0)"
    today="$(sqlite3 "${db}" \
      "SELECT COUNT(*) FROM reports WHERE date(created_at)=date('now');" \
      2>/dev/null || echo 0)"
  else
    # 防御：无 sqlite3 CLI 时走 python 兜底（同样绝不抛出，异常输出 "0 0"）
    local out
    out="$("${PYTHON_BIN}" - "${db}" 2>/dev/null <<'PY' || echo "0 0"
import sqlite3, sys
try:
    conn = sqlite3.connect(sys.argv[1])
    total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    today = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE date(created_at)=date('now')"
    ).fetchone()[0]
    conn.close()
    print(total, today)
except Exception:
    print(0, 0)
PY
)"
    out="${out:-0 0}"
    total="$(echo "${out}" | awk '{print $1}')"
    today="$(echo "${out}" | awk '{print $2}')"
  fi
  echo "统计：库内总篇数 ${total:-0}，当日新增 ${today:-0}（库：${db}）"
}

print_db_stats "${DB_PATH}"

# ── 3. 可选：全文层抓取 + 向量索引增量重建（v2，WITH_FULLTEXT=1 开启；失败不阻断退出码）──
if [ "${WITH_FULLTEXT:-0}" = "1" ]; then
  if [ -f scripts/report_fulltext.py ]; then
    echo "== 全文层抓取（--days ${DAYS}）=="
    if ! "${PYTHON_BIN}" scripts/report_fulltext.py --days "${DAYS}"; then
      echo "WARNING: 全文层抓取失败，不阻断主流程（详见上方日志）" >&2
    fi
    echo "== 向量索引增量重建 =="
    # build_index 幂等跳过已索引；缺 sentence_transformers 或模型时
    # 返回带 note 的零统计而非报错（优雅降级，不阻断）
    if ! REPORTS_DB_PATH="${DB_PATH}" "${PYTHON_BIN}" -c \
      "from agent.report_vectors import build_index; print('[daily_update] 索引:', build_index())"; then
      echo "WARNING: 向量索引重建失败，不阻断主流程（详见上方日志）" >&2
    fi
  else
    echo "提示：scripts/report_fulltext.py 不存在（v2 全文层未就位），跳过 WITH_FULLTEXT 步骤"
  fi
fi

echo "== 完成 =="
