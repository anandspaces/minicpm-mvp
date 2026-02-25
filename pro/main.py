#!/usr/bin/env python
"""
MiniCPM-o 4.5 — FastAPI + WebSocket app (vision mode).

Modes: Photo Chat (single/multi-image), Video Chat (3D-Resampler temporal_ids),
       Live Video Chat (webcam streaming with rolling frame buffer).

Model card: https://huggingface.co/openbmb/MiniCPM-o-4_5
Requires:   transformers==4.51.0  (per model card — other versions break streaming)

Run:
    uv run python pro/main.py --model-path ./models/MiniCPM-o-4_5 --device cuda --dtype bf16 --port 8084
"""

# ---------------------------------------------------------------------------
# Step 0 — Patch modeling_minicpmo.py before it is imported.
#
# With transformers != 4.51.0 two things break inside chat(stream=True):
#   a) generate() returns a plain dict  →  outputs.sequences raises AttributeError
#   b) num_beams=3 is hard-coded        →  ValueError (beam search + streamer)
#
# We locate the cached source file and fix both in-place so the official
# stream=True path works.  When the file is not found we gracefully fall back
# to stream=False (non-streaming) which always works correctly.
# ---------------------------------------------------------------------------
import glob
import os
import re

_STREAMING_PATCHED = False


def _patch_modeling_file(mp: str) -> bool:
    """Patch the cached modeling_minicpmo.py.  Returns True if OK to stream."""
    basename = os.path.basename(os.path.normpath(mp))
    hf_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    candidates = glob.glob(
        os.path.join(hf_cache, "**", "modeling_minicpmo.py"), recursive=True
    )
    candidates.append(os.path.join(hf_cache, basename, "modeling_minicpmo.py"))

    target = next((c for c in candidates if os.path.isfile(c)), None)
    if target is None:
        return False

    with open(target, "r", encoding="utf-8") as fh:
        src = fh.read()

    original = src

    # Fix (a): outputs.sequences — handle both ModelOutput and plain dict
    src = re.sub(
        r"(generated_ids\s*=\s*)outputs\.sequences(\[0\])",
        r"\1(outputs['sequences'] if isinstance(outputs, dict) else outputs.sequences)\2",
        src,
    )

    # Fix (b): num_beams=3 breaks streaming; force greedy (num_beams=1)
    src = re.sub(r"\bnum_beams\s*=\s*3\b", "num_beams=1", src)

    if src == original:
        return True  # already patched / pattern not found — streaming may still work

    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(src)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# HuggingFace shim — Gradio 4.x imports HfFolder which was removed in newer hub
# ---------------------------------------------------------------------------
import huggingface_hub

if not hasattr(huggingface_hub, "HfFolder"):

    class _HfFolder:
        @staticmethod
        def get_token():
            return getattr(huggingface_hub, "get_token", lambda: None)()

        @staticmethod
        def save_token(token):
            getattr(huggingface_hub, "login", lambda **_: None)(token=token)

        @staticmethod
        def delete_token():
            getattr(huggingface_hub, "logout", lambda: None)()

    huggingface_hub.HfFolder = _HfFolder

if not hasattr(huggingface_hub, "whoami"):

    def _whoami():
        try:
            return getattr(huggingface_hub, "get_token", lambda: None)() and True or {}
        except Exception:
            return {}

    huggingface_hub.whoami = _whoami

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import asyncio
import base64
import collections
import io
import json
import logging
import math
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn

# Patch: transformers smart_apply calls _initialize_weights on every nn.Module
# submodule, but MiniCPM-o's Resampler is a plain nn.Module without that method.
if not hasattr(nn.Module, "_initialize_weights"):
    nn.Module._initialize_weights = lambda *args: None

from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

# Patch: MiniCPM-V _tied_weights_keys → all_tied_weights_keys
if hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
    _orig_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _patched_adjust_tied_keys(self, missing_keys):
        tied = getattr(self, "all_tied_weights_keys", None) or getattr(
            self, "_tied_weights_keys", None
        )
        if tied is None or not isinstance(tied, dict):
            self.all_tied_weights_keys = {}
        elif not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = tied
        _orig_adjust_tied(self, missing_keys)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust_tied_keys

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 vision app")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument(
    "--model-path",
    type=str,
    default="openbmb/MiniCPM-o-4_5",
    help="HF model id or local path",
)
parser.add_argument("--port", type=int, default=8084)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--max-frames", type=int, default=16)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
)
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("app")

# Patch the modeling file BEFORE the model is loaded
_STREAMING_PATCHED = _patch_modeling_file(args.model_path)
log.info(
    "Modeling file patch: %s",
    "applied — streaming enabled" if _STREAMING_PATCHED else "file not found — non-streaming fallback active",
)

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------
DEVICE = args.device
if DEVICE == "cuda" and not torch.cuda.is_available():
    log.warning("CUDA not available, falling back to CPU")
    DEVICE = "cpu"
DTYPE = torch.bfloat16 if (args.dtype == "bf16" and DEVICE == "cuda") else torch.float16
log.info("Device=%s  dtype=%s", DEVICE, DTYPE)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_path = args.model_path
looks_local = model_path.startswith(("/", "./", "../"))
local_only = looks_local and os.path.isdir(model_path)
if looks_local and not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Local model path does not exist: {model_path}\n"
        "Download with: python scripts/download_minicpm_o_4_5.py --local-dir ./models/MiniCPM-o-4_5\n"
        "Or use: --model-path openbmb/MiniCPM-o-4_5"
    )

log.info("Loading MiniCPM-o 4.5 from %s …", model_path)
load_kwargs: dict = dict(
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=DTYPE,
    init_vision=True,
    init_audio=True,
    init_tts=True,
)
if local_only:
    load_kwargs["local_files_only"] = True

model = AutoModel.from_pretrained(model_path, **load_kwargs)
model = model.to(device=DEVICE).eval()
model.init_tts()

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=local_only
)
log.info("Model loaded.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 3
TIME_SCALE = 0.1
MAX_IMAGE_SIZE = 448 * 16

# Model card uses do_sample / temperature, not the old "sampling" kwarg
DEFAULT_CONFIG: Dict[str, Any] = {
    "enable_thinking": False,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 100,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "system_prompt": "",
    "max_frames": args.max_frames,
    "video_fps": 3,
    "use_tts": False,
}

# ---------------------------------------------------------------------------
# Image / video helpers
# ---------------------------------------------------------------------------

def resize_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        w, h = image.size
        if w > h:
            new_w, new_h = MAX_IMAGE_SIZE, int(h * MAX_IMAGE_SIZE / w)
        else:
            new_h, new_w = MAX_IMAGE_SIZE, int(w * MAX_IMAGE_SIZE / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image


def _map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def _group_array(arr, size):
    return [arr[i : i + size] for i in range(0, len(arr), size)]


def _uniform_sample(lst, n):
    gap = len(lst) / n
    return [lst[int(i * gap + gap / 2)] for i in range(n)]


def encode_video_file(
    video_path: str, choose_fps: int = 3, force_packing: Optional[int] = None
):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(
            min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration)
        )
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = np.array(
        _uniform_sample(list(range(len(vr))), choose_frames)
    )
    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)

    log.info(
        "Video %s: duration=%.1fs  frames=%d  packing=%d",
        video_path, video_duration, len(frame_idx), packing_nums,
    )

    raw_frames = vr.get_batch(frame_idx).asnumpy()
    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration + TIME_SCALE, TIME_SCALE)
    frame_ts_id = _map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    frames = [Image.fromarray(f.astype("uint8")).convert("RGB") for f in raw_frames]
    temporal_ids = _group_array(frame_ts_id, packing_nums)
    return (
        frames,
        temporal_ids,
        {
            "duration": round(video_duration, 1),
            "frame_count": len(frames),
            "packing": packing_nums,
        },
    )


def build_temporal_ids_for_live_frames(n_frames: int, interval_ms: int = 200):
    if n_frames == 0:
        return []
    ts_seconds = np.arange(n_frames) * (interval_ms / 1000.0)
    scale = np.arange(0, ts_seconds[-1] + TIME_SCALE * 2, TIME_SCALE)
    ts_ids = _map_to_nearest_scale(ts_seconds, scale) / TIME_SCALE
    ts_ids = ts_ids.astype(np.int32)
    packing = min(max(1, n_frames // 10), MAX_NUM_PACKING)
    return _group_array(ts_ids, packing)


# ---------------------------------------------------------------------------
# Inference
#
# The model card shows two calling patterns:
#
#   1. Simple image:
#      res = model.chat(msgs=msgs, use_tts_template=False)
#
#   2. Video:
#      res = model.chat(msgs=msgs, max_new_tokens=128, use_image_id=False,
#                       max_slice_nums=1, use_tts_template=False)
#
# stream=True is part of the MiniCPM-V API but REQUIRES transformers==4.51.0.
# We try streaming first (if the modeling file was patched), then fall back
# to non-streaming which always returns a plain string.
# ---------------------------------------------------------------------------

def _build_msgs(images: List[Image.Image], context: List[Dict]) -> List[Dict]:
    msgs = []
    for i, turn in enumerate(context):
        if turn["role"] == "user":
            text = turn["content"]
            if i == 0 and images:
                msgs.append({"role": "user", "content": images + [text]})
            else:
                msgs.append({"role": "user", "content": [text]})
        else:
            msgs.append({"role": "assistant", "content": [turn["content"]]})
    return msgs


def _build_msgs_audio(
    images: List[Image.Image], context_plus: List[Dict]
) -> List[Dict]:
    """Build msgs when the last user turn is audio (content is list of numpy array)."""
    msgs = []
    for i, turn in enumerate(context_plus):
        if turn["role"] == "user":
            content = turn["content"]
            if not isinstance(content, list):
                content = [content]
            if i == 0 and images:
                content = images + content
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "assistant", "content": [turn["content"]]})
    return msgs


def _base_kwargs(config: Dict, use_tts_template: bool = False) -> Dict[str, Any]:
    """Common kwargs for model.chat() regardless of stream mode."""
    kw: Dict[str, Any] = dict(
        tokenizer=tokenizer,
        enable_thinking=config.get("enable_thinking", False),
        use_tts_template=use_tts_template,
        max_new_tokens=config.get("max_new_tokens", 2048),
    )
    sp = config.get("system_prompt", "")
    if sp:
        kw["system_prompt"] = sp
    if config.get("do_sample", True):
        kw.update(
            do_sample=True,
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 100),
            repetition_penalty=config.get("repetition_penalty", 1.05),
        )
    else:
        kw["do_sample"] = False
    return kw


def _video_kwargs(images: List, temporal_ids, is_video: bool) -> Dict[str, Any]:
    """Extra kwargs for video / live-frame inference."""
    if not is_video:
        return {}
    if temporal_ids:
        return {"temporal_ids": temporal_ids, "use_image_id": False, "max_slice_nums": 1}
    n = len(images)
    return {
        "max_inp_length": 4352,
        "use_image_id": False,
        "max_slice_nums": 1 if n > 16 else 2,
    }


def infer_stream(images, context, config, temporal_ids, is_video) -> Iterator[str]:
    """model.chat(stream=True) — requires patched modeling file."""
    msgs = _build_msgs(images, context)
    kw = _base_kwargs(config)
    kw.update(_video_kwargs(images, temporal_ids, is_video))
    kw["stream"] = True
    gen = model.chat(msgs=msgs, **kw)
    for token in gen:
        yield token


def infer_stream_with_msgs(msgs, images, config, temporal_ids, is_video) -> Iterator[str]:
    """Stream inference with pre-built msgs (e.g. for audio input); use_tts_template=True."""
    kw = _base_kwargs(config, use_tts_template=True)
    kw.update(_video_kwargs(images, temporal_ids, is_video))
    kw["stream"] = True
    gen = model.chat(msgs=msgs, **kw)
    for token in gen:
        yield token


def infer_nonstream_with_msgs(msgs, images, config, temporal_ids, is_video) -> str:
    """Non-stream inference with pre-built msgs; use_tts_template=True."""
    kw = _base_kwargs(config, use_tts_template=True)
    kw.update(_video_kwargs(images, temporal_ids, is_video))
    result = model.chat(msgs=msgs, **kw)
    if isinstance(result, (list, tuple)):
        return str(result[-1])
    return str(result) if result is not None else ""


def infer_nonstream(images, context, config, temporal_ids, is_video) -> str:
    """model.chat() without streaming — always works, returns full string."""
    msgs = _build_msgs(images, context)
    kw = _base_kwargs(config)
    kw.update(_video_kwargs(images, temporal_ids, is_video))
    result = model.chat(msgs=msgs, **kw)
    if isinstance(result, (list, tuple)):
        return str(result[-1])   # thinking mode returns (thinking, answer)
    return str(result) if result is not None else ""


def infer_tts(images, context, config, temporal_ids, is_video) -> tuple:
    """Synthesize last assistant reply to WAV. Returns (wav_bytes, sample_rate) or (None, 0) on failure."""
    try:
        msgs = _build_msgs(images, context)
        kw = _base_kwargs(config)
        kw["use_tts_template"] = True
        kw["generate_audio"] = True
        kw.update(_video_kwargs(images, temporal_ids, is_video))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            model.chat(msgs=msgs, output_audio_path=out_path, **kw)
            with open(out_path, "rb") as f:
                return (f.read(), 24000)
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
    except Exception as e:
        log.exception("TTS synthesis failed: %s", e)
        return (None, 0)


def _clean(text: str) -> str:
    text = re.sub(r"<box>.*?</box>", "", text, flags=re.DOTALL)
    return text.replace("<ref>", "").replace("</ref>", "").replace("<box>", "").replace("</box>", "").strip()


# ---------------------------------------------------------------------------
# Async inference bridge
# Inference runs in a thread; tokens are pushed to an asyncio.Queue
# so the WebSocket handler can forward them without blocking the event loop.
# ---------------------------------------------------------------------------

async def run_async(images, context, config, temporal_ids, is_video, queue: asyncio.Queue):
    loop = asyncio.get_event_loop()

    def _worker():
        try:
            if _STREAMING_PATCHED:
                try:
                    for tok in infer_stream(images, context, config, temporal_ids, is_video):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", tok))
                    return
                except Exception as e:
                    log.warning("stream=True failed (%s), retrying without streaming", e)

            # Non-streaming fallback
            result = infer_nonstream(images, context, config, temporal_ids, is_video)
            loop.call_soon_threadsafe(queue.put_nowait, ("token", result))

        except Exception as e:
            log.exception("Inference error: %s", e)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop.run_in_executor(_executor, _worker)


async def run_async_msgs(msgs, images, config, temporal_ids, is_video, queue: asyncio.Queue):
    """Like run_async but with pre-built msgs (for audio input); uses use_tts_template=True."""
    loop = asyncio.get_event_loop()

    def _worker():
        try:
            if _STREAMING_PATCHED:
                try:
                    for tok in infer_stream_with_msgs(
                        msgs, images, config, temporal_ids, is_video
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", tok))
                    return
                except Exception as e:
                    log.warning("stream=True (audio) failed (%s), retrying", e)

            result = infer_nonstream_with_msgs(
                msgs, images, config, temporal_ids, is_video
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("token", result))

        except Exception as e:
            log.exception("Inference error (audio): %s", e)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop.run_in_executor(_executor, _worker)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="MiniCPM-o 4.5", version="3.0.0")
_executor = ThreadPoolExecutor(max_workers=2)
PRO_DIR = Path(__file__).resolve().parent


@app.get("/")
async def index():
    return FileResponse(PRO_DIR / "live.html", media_type="text/html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MiniCPM-o-4.5",
        "device": DEVICE,
        "streaming_patched": _STREAMING_PATCHED,
    }


def _new_session() -> Dict[str, Any]:
    return {
        "images": [],
        "video_frames": [],
        "video_temporal_ids": None,
        "video_info": {},
        "frame_buffer": collections.deque(maxlen=args.max_frames),
        "frame_timestamps": collections.deque(maxlen=args.max_frames),
        "context": [],
        "mode": "photo",
        "config": dict(DEFAULT_CONFIG),
        "lock": threading.Lock(),
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = _new_session()

    async def send(event: str, data: Optional[Dict] = None):
        payload: Dict[str, Any] = {"event": event}
        if data is not None:
            payload["data"] = data
        await websocket.send_json(payload)

    await send(
        "session_ready",
        {
            "max_frames": args.max_frames,
            "config": session["config"],
            "streaming": _STREAMING_PATCHED,
        },
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send("error", {"message": "Invalid JSON"})
                continue
            if not isinstance(msg, dict):
                await send("error", {"message": "Expected JSON object"})
                continue

            action = msg.get("action")
            if not action:
                await send("error", {"message": "Missing 'action'"})
                continue

            # ── upload_images ─────────────────────────────────────
            if action == "upload_images":
                b64_list = msg.get("images", [])
                if not b64_list:
                    await send("error", {"message": "No images provided"})
                    continue
                await send("upload_started", {"type": "image", "count": len(b64_list)})
                try:
                    imgs = []
                    for b64 in b64_list:
                        data = base64.b64decode(b64)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        imgs.append(resize_image(img))
                    session.update(
                        images=imgs, video_frames=[], video_temporal_ids=None,
                        context=[], mode="photo"
                    )
                    await send("upload_done", {"type": "image", "count": len(imgs)})
                except Exception as e:
                    log.exception("upload_images failed")
                    await send("error", {"message": str(e)})
                continue

            # ── upload_video ──────────────────────────────────────
            if action == "upload_video":
                b64_data = msg.get("data")
                if not b64_data:
                    await send("error", {"message": "No video data"})
                    continue
                fps = msg.get("fps", session["config"].get("video_fps", 3))
                await send("upload_started", {"type": "video"})
                try:
                    raw_bytes = base64.b64decode(b64_data)
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        f.write(raw_bytes)
                        tmp_path = f.name
                    try:
                        frames, temporal_ids, info = encode_video_file(tmp_path, choose_fps=fps)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    session.update(
                        video_frames=frames, video_temporal_ids=temporal_ids,
                        video_info=info, images=[], context=[], mode="video"
                    )
                    await send("upload_done", {"type": "video", **info})
                except Exception as e:
                    log.exception("upload_video failed")
                    await send("error", {"message": str(e)})
                continue

            # ── chat_with_audio (user message as audio; model STT + reply) ──
            if action == "chat_with_audio":
                audio_b64 = msg.get("audio_b64")
                if not audio_b64:
                    await send("error", {"message": "No audio_b64 provided"})
                    continue
                mode = msg.get("mode", session["mode"])
                session["mode"] = mode

                try:
                    raw = base64.b64decode(audio_b64)
                    audio_np, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
                except Exception as e:
                    log.exception("chat_with_audio decode failed: %s", e)
                    await send("error", {"message": f"Invalid audio: {e}"})
                    continue

                if mode == "photo":
                    media = session["images"]
                    temporal_ids = None
                    is_video = False
                elif mode == "video":
                    media = session["video_frames"]
                    temporal_ids = session["video_temporal_ids"]
                    is_video = True
                elif mode == "live":
                    with session["lock"]:
                        media = list(session["frame_buffer"])
                    interval_ms = 200
                    if len(session["frame_timestamps"]) >= 2:
                        ts = list(session["frame_timestamps"])
                        interval_ms = int((ts[-1] - ts[0]) / (len(ts) - 1) * 1000)
                    temporal_ids = build_temporal_ids_for_live_frames(len(media), interval_ms)
                    is_video = True
                else:
                    media = []
                    temporal_ids = None
                    is_video = False

                context_plus = list(session["context"]) + [
                    {"role": "user", "content": [audio_np]}
                ]
                msgs = _build_msgs_audio(media, context_plus)
                session["context"].append({"role": "user", "content": "[voice message]"})

                await send("chat_started", {"frame_count": len(media), "mode": mode})

                queue_audio: asyncio.Queue = asyncio.Queue()
                await run_async_msgs(
                    msgs, media, dict(session["config"]),
                    temporal_ids, is_video, queue_audio
                )

                full_text = ""
                error_occurred = False
                try:
                    while True:
                        kind, payload = await asyncio.wait_for(queue_audio.get(), timeout=180)
                        if kind == "token":
                            full_text += payload
                            await send("chat_token", {"text": payload})
                        elif kind == "error":
                            await send("error", {"message": payload})
                            error_occurred = True
                            break
                        elif kind == "done":
                            break
                except asyncio.TimeoutError:
                    await send("error", {"message": "Generation timed out (180 s)"})
                    error_occurred = True
                except WebSocketDisconnect:
                    session["context"].pop()
                    raise

                if error_occurred:
                    session["context"].pop()
                    continue

                full_text = _clean(full_text)
                session["context"].append({"role": "assistant", "content": full_text})
                await send("chat_done", {"text": full_text})

                if session["config"].get("use_tts"):
                    loop = asyncio.get_event_loop()
                    try:
                        audio_bytes, sr = await loop.run_in_executor(
                            _executor,
                            lambda: infer_tts(
                                media,
                                list(session["context"]),
                                dict(session["config"]),
                                temporal_ids,
                                is_video,
                            ),
                        )
                        if audio_bytes:
                            await send(
                                "chat_audio",
                                {
                                    "audio_b64": base64.b64encode(audio_bytes).decode(),
                                    "sample_rate": sr,
                                },
                            )
                    except Exception as e:
                        log.exception("TTS failed: %s", e)
                continue

            # ── frame (live webcam) ───────────────────────────────
            if action == "frame":
                b64_data = msg.get("data")
                if not b64_data:
                    continue
                try:
                    raw_bytes = base64.b64decode(b64_data)
                    img = resize_image(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                    with session["lock"]:
                        session["frame_buffer"].append(img)
                        session["frame_timestamps"].append(time.time())
                except Exception:
                    pass
                continue

            # ── chat / regenerate ─────────────────────────────────
            if action in ("chat", "regenerate"):
                if action == "chat":
                    text = (msg.get("text") or "").strip()
                    if not text:
                        await send("error", {"message": "Empty message"})
                        continue
                    mode = msg.get("mode", session["mode"])
                    session["mode"] = mode
                else:
                    if len(session["context"]) < 2:
                        await send("error", {"message": "Nothing to regenerate"})
                        continue
                    session["context"].pop()
                    text = session["context"].pop()["content"]
                    mode = session["mode"]

                # Resolve media
                if mode == "photo":
                    media = session["images"]
                    temporal_ids = None
                    is_video = False
                elif mode == "video":
                    media = session["video_frames"]
                    temporal_ids = session["video_temporal_ids"]
                    is_video = True
                elif mode == "live":
                    with session["lock"]:
                        media = list(session["frame_buffer"])
                    if not media:
                        await send("error", {"message": "No webcam frames yet"})
                        continue
                    interval_ms = 200
                    if len(session["frame_timestamps"]) >= 2:
                        ts = list(session["frame_timestamps"])
                        interval_ms = int((ts[-1] - ts[0]) / (len(ts) - 1) * 1000)
                    temporal_ids = build_temporal_ids_for_live_frames(len(media), interval_ms)
                    is_video = True
                else:
                    await send("error", {"message": f"Unknown mode: {mode}"})
                    continue

                if not media:
                    await send("error", {"message": "No media uploaded. Please upload an image or video first."})
                    continue

                session["context"].append({"role": "user", "content": text})
                await send("chat_started", {"frame_count": len(media), "mode": mode})

                queue: asyncio.Queue = asyncio.Queue()
                await run_async(
                    media, list(session["context"]),
                    dict(session["config"]), temporal_ids, is_video, queue
                )

                full_text = ""
                error_occurred = False
                try:
                    while True:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=180)
                        if kind == "token":
                            full_text += payload
                            await send("chat_token", {"text": payload})
                        elif kind == "error":
                            await send("error", {"message": payload})
                            error_occurred = True
                            break
                        elif kind == "done":
                            break
                except asyncio.TimeoutError:
                    await send("error", {"message": "Generation timed out (180 s)"})
                    error_occurred = True
                except WebSocketDisconnect:
                    session["context"].pop()
                    raise

                if error_occurred:
                    session["context"].pop()
                    continue

                full_text = _clean(full_text)
                session["context"].append({"role": "assistant", "content": full_text})
                await send("chat_done", {"text": full_text})

                if session["config"].get("use_tts"):
                    loop = asyncio.get_event_loop()
                    try:
                        audio_bytes, sr = await loop.run_in_executor(
                            _executor,
                            lambda: infer_tts(
                                media,
                                list(session["context"]),
                                dict(session["config"]),
                                temporal_ids,
                                is_video,
                            ),
                        )
                        if audio_bytes:
                            await send(
                                "chat_audio",
                                {
                                    "audio_b64": base64.b64encode(audio_bytes).decode(),
                                    "sample_rate": sr,
                                },
                            )
                    except Exception as e:
                        log.exception("TTS failed: %s", e)
                continue

            # ── set_config ────────────────────────────────────────
            if action == "set_config":
                for k, v in msg.get("config", {}).items():
                    if k in session["config"]:
                        session["config"][k] = v
                max_f = session["config"].get("max_frames", args.max_frames)
                if max_f != session["frame_buffer"].maxlen:
                    old = list(session["frame_buffer"])
                    old_ts = list(session["frame_timestamps"])
                    session["frame_buffer"] = collections.deque(old[-max_f:], maxlen=max_f)
                    session["frame_timestamps"] = collections.deque(old_ts[-max_f:], maxlen=max_f)
                await send("config_updated", {"config": session["config"]})
                continue

            # ── clear ─────────────────────────────────────────────
            if action == "clear":
                session["context"] = []
                await send("cleared", {})
                continue

            await send("error", {"message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        try:
            await send("error", {"message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting MiniCPM-o 4.5 on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())