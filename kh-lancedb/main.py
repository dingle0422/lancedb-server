"""项目根目录启动入口。

支持直接执行：
    python main.py
"""

from __future__ import annotations

import importlib

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
