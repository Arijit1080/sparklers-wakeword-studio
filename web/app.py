"""Wakeword Studio — FastAPI web UI.

Two pages:
    /           home — pick which models to listen for, start listening
    /dashboard  live trigger feed + per-model score bars
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.state import SERVICE

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Sparklers Wakeword Studio")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.on_event("startup")
async def _startup() -> None:
    SERVICE.attach_loop(asyncio.get_event_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    SERVICE.shutdown()


# ---- pages ----

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="home.html",
        context={"models": SERVICE.list_models(),
                 "status": SERVICE.status},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context={"status": SERVICE.status,
                 "triggers": SERVICE.triggers()},
    )


@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="train.html",
        context={"status": SERVICE.status, "models": SERVICE.list_models()},
    )


@app.post("/api/train/start")
async def api_train_start(
    keyword: str = Form(...),
    n_per_voice: int = Form(50),
    neg_per_voice: int = Form(80),
):
    return JSONResponse(SERVICE.start_training(
        keyword=keyword, n_per_voice=n_per_voice, neg_per_voice=neg_per_voice,
    ))


@app.post("/api/train/record_samples")
async def api_record_samples(
    keyword: str = Form(...),
    n_samples: int = Form(10),
):
    """Optional 'record yourself' flow before training.

    Records `n_samples` short clips of the user saying the keyword;
    each clip is saved under data/train/user_pos/<kw_safe>/ and gets
    mixed into the next train run as pitch-shifted positives.  Returns
    immediately; the actual recording runs in a background thread —
    poll /api/status or listen on /events for progress.
    """
    return JSONResponse(SERVICE.start_recording_samples(
        keyword=keyword, n_samples=n_samples,
    ))


@app.post("/api/train/stop_recording")
async def api_stop_recording():
    return JSONResponse(SERVICE.stop_recording_samples())


@app.post("/api/train/clear_user_samples")
async def api_clear_user_samples(keyword: str = Form(...)):
    return JSONResponse(SERVICE.clear_user_samples(keyword=keyword))


@app.get("/api/train/user_samples")
async def api_user_samples_count(keyword: str):
    return JSONResponse({
        "ok": True, "keyword": keyword,
        "count": SERVICE.count_user_samples(keyword=keyword),
    })


@app.post("/api/model/delete")
async def api_model_delete(name: str = Form(...)):
    return JSONResponse(SERVICE.delete_model(name))


# ---- API ----

@app.post("/api/listen/start")
async def api_start(
    models: str = Form(""),
    threshold: float = Form(0.3),
    patience: int = Form(1),
    vad_threshold: float = Form(0.3),
):
    mlist = [m.strip() for m in models.split(",") if m.strip()] or None
    return JSONResponse(SERVICE.start_listening(
        models=mlist, threshold=threshold, patience=patience,
        vad_threshold=vad_threshold,
    ))


@app.post("/api/listen/stop")
async def api_stop():
    return JSONResponse(SERVICE.stop_listening())


@app.get("/api/status")
async def api_status():
    from dataclasses import asdict
    return {"status": asdict(SERVICE.status),
            "models": SERVICE.list_models(),
            "triggers": SERVICE.triggers()}


@app.get("/events")
async def sse_events():
    q = SERVICE.subscribe()
    async def stream():
        try:
            while True:
                evt = await q.get()
                yield f"event: {evt.get('type','message')}\ndata: {json.dumps(evt)}\n\n"
        finally:
            SERVICE.unsubscribe(q)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
