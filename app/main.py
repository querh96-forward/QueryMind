from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import router, service
from app.config import get_settings


settings = get_settings()
ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(
    title="QueryMind",
    description="基于LangGraph的可治理企业数据智能分析平台",
    version=settings.app_version,
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.middleware("http")
async def process_time(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - started) * 1000, 2))
    return response


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    if settings.debug:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "服务暂时无法处理请求，请稍后重试"})


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "static" / "index.html")
