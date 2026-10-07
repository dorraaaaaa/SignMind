"""
Web interface server for ASL Translator.

Endpoints
---------
GET  /         → HTML frontend
GET  /video    → MJPEG live camera stream (annotated)
GET  /events   → Server-Sent Events with gesture data (~10 fps)
POST /speak    → Trigger TTS for the current sentence
POST /clear    → Clear the sentence

Usage  (run from project root)
-----
    python web/server.py
    → Open http://localhost:8000 in your browser
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.predictor import GesturePredictor
from core.tts import TextToSpeech


# ── Shared camera stream ──────────────────────────────────────────────────────

class CameraStream:
    """
    Runs camera capture + inference in a background thread so both the
    MJPEG endpoint and the SSE endpoint can consume the same frame without
    opening two camera connections.
    """

    def __init__(self, predictor: GesturePredictor, camera_index: int = 0):
        self._predictor = predictor
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Shared state (protected by _lock)
        self._frame:      Optional[np.ndarray] = None
        self._gesture:    Optional[str]        = None
        self._confidence: float                = 0.0
        self._word:       str                  = ""
        self._sentence:   str                  = ""
        self._lock = threading.Lock()

        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            frame, gesture, conf, word, sentence = self._predictor.process_frame(frame)

            with self._lock:
                self._frame      = frame
                self._gesture    = gesture
                self._confidence = conf
                self._word       = word
                self._sentence   = sentence

    # ------------------------------------------------------------------

    def get_jpeg(self) -> Optional[bytes]:
        """Return latest annotated frame as a JPEG byte string."""
        with self._lock:
            if self._frame is None:
                return None
            _, buf = cv2.imencode(
                ".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            return buf.tobytes()

    def get_state(self) -> dict:
        """Return current gesture state as a plain dict."""
        with self._lock:
            return {
                "gesture":    self._gesture,
                "confidence": round(self._confidence, 3),
                "word":       self._word,
                "sentence":   self._sentence,
            }

    def clear(self) -> None:
        self._predictor.clear()

    def release(self) -> None:
        self._running = False
        time.sleep(0.1)
        self._cap.release()
        self._predictor.release()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="ASL Translator", docs_url=None, redoc_url=None)

    predictor = GesturePredictor(config)
    tts       = TextToSpeech(config["tts"]["language"])
    stream    = CameraStream(predictor, config["data"]["camera_index"])

    STATIC = Path(__file__).parent / "static"

    # ── Routes ────────────────────────────────────────────────────────────

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    # ── MJPEG camera stream ────────────────────────────────────────────────
    @app.get("/video")
    async def video():
        def frames():
            while True:
                jpeg = stream.get_jpeg()
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                time.sleep(1 / 30)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── Server-Sent Events (gesture data) ─────────────────────────────────
    @app.get("/events")
    async def events():
        async def stream_events():
            while True:
                data = stream.get_state()
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "Connection":       "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Actions ────────────────────────────────────────────────────────────
    @app.post("/speak")
    async def speak():
        state    = stream.get_state()
        sentence = state.get("sentence", "")
        if sentence:
            tts.speak(sentence)
        return JSONResponse({"ok": True, "sentence": sentence})

    @app.post("/clear")
    async def clear():
        stream.clear()
        return JSONResponse({"ok": True})

    # ── Cleanup ────────────────────────────────────────────────────────────
    @app.on_event("shutdown")
    def shutdown():
        stream.release()
        tts.cleanup()

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 52)
    print("  ASL TRANSLATOR  —  Web Interface")
    print("=" * 52)

    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config.yaml not found at {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    model_path = ROOT / "models" / "gesture_model.pkl"
    if not model_path.exists():
        print("[ERROR] Trained model not found.")
        print("        Train first:  python main.py  → option 2")
        sys.exit(1)

    print("\n  ✓  Model found")
    print("  ✓  Starting server …")
    print("\n  Open in browser →  http://localhost:8000\n")
    print("  Press Ctrl+C to stop\n")

    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")