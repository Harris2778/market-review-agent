#!/bin/sh
# 生产容器入口：卷属主修正 + 降权启动。
#
# 背景：Dockerfile 构建期 chown 的 /data 会被平台挂载的卷整个遮住，
# 而 Railway/Render 挂载的卷默认 root:root，agent 用户无权写入
# （症状：写 /data/*.uploading 报 Permission denied）。
# 因此容器以 root 启动，先在运行期把挂载点属主修正为 agent，
# 再降权执行主进程（优先 setpriv，干净 exec 不 fork；兜底 su）。
set -e

chown -R agent:agent /data 2>/dev/null || true

if command -v setpriv >/dev/null 2>&1; then
  exec setpriv --reuid=agent --regid=agent --clear-groups python main.py
fi
exec su -s /bin/sh agent -c "exec python main.py"
