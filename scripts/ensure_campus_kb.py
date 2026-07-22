"""生产启动期校园知识库落位（幂等）。

背景：campus_kb.db（约 93MB SQLite，44,332 条清华经管校园知识库数据）
是本地数据文件，不入 git 的 data/；Railway 卷 /data 初始为空，生产端
校园知识库因此无数据可用。

方案：镜像内携带 gzip 压缩快照 assets/campus_kb.db.gz（约 30MB，
随仓库提交），容器启动入口（scripts/docker-entrypoint.sh）在降权前
调用本模块——若 $DATA_DIR/campus_kb.db 不存在则解压落位；已存在则
原样保留（幂等，用户后续手动更新的数据不会被快照覆盖）。

环境变量：
  DATA_DIR          数据目录，默认 /data（与 Dockerfile 一致）
  CAMPUS_KB_ASSET   覆盖压缩快照路径（测试用），默认仓库内 assets/campus_kb.db.gz
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_ASSET = Path(__file__).resolve().parent.parent / "assets" / "campus_kb.db.gz"


def ensure_campus_kb(data_dir, asset_path=DEFAULT_ASSET):
    """确保 data_dir/campus_kb.db 可用，返回 (status, detail)。

    status 取值：
      "ready"    —— db 已存在且非空，原样保留（幂等，不覆盖）
      "restored" —— db 缺失，已从 gzip 快照解压落位
      "skipped"  —— db 缺失且快照不存在，无法落位（不视为错误）
    """
    data_dir = Path(data_dir)
    asset_path = Path(asset_path)
    target = data_dir / "campus_kb.db"

    if target.exists() and target.stat().st_size > 0:
        return ("ready", str(target))

    if not asset_path.exists():
        return ("skipped", f"asset missing: {asset_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    # 先解压到同目录临时文件再原子替换，避免中途失败留下半截 db
    fd, tmp = tempfile.mkstemp(dir=data_dir, prefix=".campus_kb.", suffix=".tmp")
    os.close(fd)
    try:
        with gzip.open(asset_path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return ("restored", str(target))


def main() -> int:
    data_dir = os.environ.get("DATA_DIR", "/data")
    asset = os.environ.get("CAMPUS_KB_ASSET") or DEFAULT_ASSET
    try:
        status, detail = ensure_campus_kb(data_dir, asset)
    except Exception as exc:  # 落位失败不阻塞主服务启动
        print(f"[campus_kb] restore failed: {exc}", file=sys.stderr)
        return 0
    print(f"[campus_kb] {status}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
