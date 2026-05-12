"""项目根目录启动入口。

支持直接执行：
    python main.py
"""

from __future__ import annotations

import importlib
import os

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn = importlib.import_module("uvicorn")

    # 反向代理前缀：浏览器访问 https://your.domain/<PROXY_ROOT_PATH>/...
    # 代理把 PROXY_ROOT_PATH 剥掉后转给本服务。把它作为 ASGI root_path 传给 uvicorn，
    # FastAPI / Gradio 在生成 URL（/openapi.json、Gradio 的 assets / queue / info）时会自动加回前缀，
    # 浏览器请求才能再次穿过代理。本地直连留空字符串即可。
    root_path = os.environ.get("PROXY_ROOT_PATH", "/kh-lancedb").rstrip("/")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
        root_path=root_path,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
