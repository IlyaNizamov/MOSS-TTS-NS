"""
MOSS-TTS HTTP server.

Запуск:
    python server.py --config configs/llama_cpp/default.yaml
    python server.py --config configs/llama_cpp/default.yaml --host 0.0.0.0 --port 8080

Фиксация голоса:
    # Через reference audio (лучший способ — клонирование голоса):
    python server.py --config configs/llama_cpp/default.yaml --reference voice.wav

    # Через seed (один и тот же seed = один и тот же голос):
    python server.py --config configs/llama_cpp/default.yaml --seed 42

Примеры запросов:

    # Получить WAV файл
    curl -X POST http://localhost:8000/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello, world!"}' \
        --output speech.wav

    # С конкретным seed (переопределяет серверный)
    curl -X POST http://localhost:8000/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello, world!", "seed": 42}' \
        --output speech.wav

    # Получить информацию о сервере
    curl http://localhost:8000/health
"""

from __future__ import annotations

import argparse
import io
import logging
import time

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from moss_tts_delay.llama_cpp._constants import SAMPLE_RATE
from moss_tts_delay.llama_cpp.pipeline import LlamaCppPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="MOSS-TTS Server", version="1.0.0")
pipeline: LlamaCppPipeline | None = None
_reference_audio: str | None = None
_default_seed: int | None = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Текст для синтеза")
    language: str | None = Field(None, description="Язык: zh, en и др.")
    instruction: str | None = Field(None, description="Инструкция для генерации")
    quality: str | None = Field(None, description="Качество генерации")
    max_new_tokens: int | None = Field(None, ge=1, le=10000, description="Макс. шагов генерации")
    seed: int | None = Field(None, ge=0, description="Random seed для воспроизводимого голоса")
    format: str = Field("wav", description="Формат: wav или raw")


class HealthResponse(BaseModel):
    status: str
    model: str
    sample_rate: int


@app.get("/health", response_model=HealthResponse)
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return HealthResponse(
        status="ok",
        model="MOSS-TTS (llama.cpp)",
        sample_rate=SAMPLE_RATE,
    )


@app.post("/tts")
def tts(req: TTSRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    t0 = time.time()

    seed = req.seed if req.seed is not None else _default_seed
    if seed is not None:
        np.random.seed(seed)

    try:
        waveform = pipeline.generate(
            text=req.text,
            reference_audio=_reference_audio,
            instruction=req.instruction,
            quality=req.quality,
            language=req.language,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    if waveform.size == 0:
        raise HTTPException(status_code=500, detail="No audio generated")

    duration = len(waveform) / SAMPLE_RATE
    elapsed = time.time() - t0
    log.info(
        "Generated %.2fs audio in %.2fs (RTF=%.2f) for: %s",
        duration,
        elapsed,
        elapsed / duration if duration > 0 else 0,
        req.text[:80],
    )

    if req.format == "raw":
        return Response(
            content=waveform.astype(np.float32).tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Duration": f"{duration:.3f}",
                "X-Channels": "1",
                "X-Dtype": "float32",
            },
        )

    buf = io.BytesIO()
    sf.write(buf, waveform, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"X-Duration": f"{duration:.3f}"},
    )


def main():
    global pipeline, _reference_audio, _default_seed

    parser = argparse.ArgumentParser(description="MOSS-TTS HTTP Server")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--reference", default=None,
                        help="Path to reference audio WAV (24kHz) — fixes voice for all requests")
    parser.add_argument("--seed", type=int, default=None,
                        help="Default random seed — fixes voice when no reference audio is used")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    if args.n_gpu_layers is not None:
        config.n_gpu_layers = args.n_gpu_layers
    if args.low_memory:
        config.low_memory = True

    _reference_audio = args.reference
    _default_seed = args.seed

    if _reference_audio:
        log.info("Reference audio: %s (voice will be consistent across requests)", _reference_audio)
    if _default_seed is not None:
        log.info("Default seed: %d (voice will be consistent across requests)", _default_seed)

    log.info("Loading pipeline...")
    pipeline = LlamaCppPipeline(config)
    pipeline.__enter__()
    log.info("Pipeline ready. Starting server on %s:%d", args.host, args.port)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        pipeline.__exit__(None, None, None)
        pipeline = None


if __name__ == "__main__":
    main()
