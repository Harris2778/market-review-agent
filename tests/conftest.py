"""pytest 共享基础设施。

要点：
1. 把项目根目录加入 sys.path，使 `import main` / `import agent.xxx` 可用。
2. 在 **模块级**（任何测试模块 import main 之前）用 os.environ.setdefault
   注入假的环境变量——main.py 在 import 时就会校验 AGENT_API_KEY，
   缺失会直接 raise RuntimeError 拒绝启动。
   main.py 内部的 load_dotenv() 默认不覆盖已存在的环境变量，
   因此这里设置的假值优先于项目根目录 .env 中的真实密钥。
3. 提供一个 autouse 的 env fixture，用 monkeypatch 在每个测试用例级别
   再次固定这些假值，保证测试之间互不污染、且绝不触达真实 API。
"""

import os
import sys

import pytest

# ── 1. 项目根目录加入 sys.path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 2. 假的环境变量（必须在 import main 之前设置）──
FAKE_ENV = {
    "AGENT_API_KEY": "test-agent-api-key",
    "DEEPSEEK_API_KEY": "test-deepseek-api-key",
    "TUSHARE_TOKEN": "test-tushare-token",
    "FINNHUB_API_KEY": "test-finnhub-api-key",
    "FRED_API_KEY": "test-fred-api-key",
    "SINA_MCP_TOKEN": "test-sina-mcp-token",
    # 存档/图表目录隔离到 tmp，防止测试往真实 data/archive 与 charts 写垃圾
    "ARCHIVE_DIR": "/tmp/market_review_agent_test_archive",
    "CHART_DIR": "/tmp/market_review_agent_test_charts",
    "REPORTS_DB_PATH": "/tmp/market_review_agent_test_reports.db",
}

for _key, _value in FAKE_ENV.items():
    os.environ.setdefault(_key, _value)


# ── 3. autouse fixture：每个测试用例级别固定假环境变量 ──
@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """为每个测试用例固定假的 API Key/Token 环境变量。

    使用 monkeypatch，测试结束后自动恢复原值，避免用例间污染。
    所有外部调用（tushare/requests/openai/yfinance/fredapi/finnhub）
    仍需在各自测试中显式 mock——这里的假 Key 只是双保险。
    """
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    # 测试缺省视为「匿名 B 站」：真实 .env 若配置了 BILI_SESSDATA，
    # main.py 的 load_dotenv 会把它带进 os.environ，导致匿名路径的
    # 测试走错 wbi 分支；登录态测试会自行 monkeypatch.setenv 覆盖。
    monkeypatch.delenv("BILI_SESSDATA", raising=False)
    return FAKE_ENV
