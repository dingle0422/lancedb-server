"""retrieval_service 主入口。

裸启动：``uvicorn app.main:app --host 0.0.0.0 --port 5000``
容器启动：见 ``Dockerfile``，``docker-compose up`` 即可。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routers import chunks, health, meta, relations, search


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("retrieval_service")
    logger.info(
        "retrieval_service starting: store_dir=%s, scalar_index=%s, auth=%s",
        settings.store_dir,
        settings.enable_scalar_index,
        "on" if settings.api_key else "off",
    )
    yield
    logger.info("retrieval_service stopped")


app = FastAPI(
    title="page-know-how retrieval service",
    description="LanceDB 驱动的 BM25 + 向量混合检索微服务（每 policy 一张表）。",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chunks.router)
app.include_router(search.router)
app.include_router(relations.router)
app.include_router(meta.router)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/ui", include_in_schema=False)
async def web_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "ui.html")


# ---- Gradio 前端（Lance Data Viewer 风格），同进程挂载到 /gradio ----
# 反向代理前缀通过 uvicorn 的 --root-path（ASGI scope.root_path）传入，详见 main.py。
try:
    import gradio as gr  # type: ignore

    from .ui_gradio import build_demo

    app = gr.mount_gradio_app(app, build_demo(), path="/gradio")
    logging.getLogger(__name__).info("Gradio mounted at /gradio/")

except Exception as _gradio_exc:  # noqa: BLE001
    logging.getLogger(__name__).warning(
        "gradio UI 未启用（pip install gradio 后即可访问）：%s", _gradio_exc
    )
