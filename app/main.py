"""retrieval_service 主入口。

裸启动：``uvicorn app.main:app --host 0.0.0.0 --port 5000``
容器启动：见 ``Dockerfile``，``docker-compose up`` 即可。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import get_settings
from .routers import chunks, health, meta, relations, search, v2


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
    title="lancedb retrieval service",
    description="LanceDB 驱动的向量/全文混合检索服务（兼容 legacy policy API + generic collection API）。",
    version=__version__,
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
app.include_router(meta.router)
_settings = get_settings()
if _settings.enable_legacy_relations:
    app.include_router(relations.router)
if _settings.enable_generic_api:
    app.include_router(v2.router)

# ---- Gradio 前端：policy(legacy) 与 generic(v2) 双界面 ----
# 反向代理前缀通过 uvicorn 的 --root-path（ASGI scope.root_path）传入，详见 main.py。
try:
    import gradio as gr  # type: ignore

    if _settings.enable_legacy_ui:
        from .ui_gradio import build_demo

        app = gr.mount_gradio_app(app, build_demo(), path="/gradio")
        logging.getLogger(__name__).info("Policy Gradio mounted at /gradio/")

    if _settings.enable_generic_gradio and _settings.enable_generic_api:
        from .ui_gradio_generic import build_generic_demo

        app = gr.mount_gradio_app(app, build_generic_demo(), path="/gradio-generic")
        logging.getLogger(__name__).info("Generic Gradio mounted at /gradio-generic/")

except Exception as _gradio_exc:  # noqa: BLE001
    logging.getLogger(__name__).warning(
        "gradio UI 未启用（pip install gradio 后即可访问）：%s", _gradio_exc
    )
