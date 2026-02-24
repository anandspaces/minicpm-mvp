#!/usr/bin/env python
"""
MiniCPM-V 4.5 — Full-feature FastAPI + WebSocket app.

Modes: Photo Chat (single/multi-image), Video Chat (with 3D-Resampler temporal_ids),
       Live Video Chat (webcam streaming with rolling frame buffer).

Run:
    uv run python src/app.py --model-path ./models/MiniCPM-V-4_5 --device cuda --dtype bf16 --port 8083
"""

# ---------------------------------------------------------------------------
# HuggingFace shim — Gradio 4.x imports HfFolder which was removed
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
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

# ---------------------------------------------------------------------------
# Compatibility: MiniCPM-V 4.5 _tied_weights_keys → all_tied_weights_keys
# ---------------------------------------------------------------------------
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
parser = argparse.ArgumentParser(description="MiniCPM-V 4.5 Full App")
parser.add_argument("--device", type=str, default="cuda", help="cuda, mps, or cpu")
parser.add_argument("--dtype", type=str, default="bf16", help="bf16 or fp16")
parser.add_argument(
    "--model-path",
    type=str,
    default="openbmb/MiniCPM-V-4_5",
    help="HuggingFace model id or local path",
)
parser.add_argument("--port", type=int, default=8083)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--max-frames", type=int, default=16, help="Max live-video frames kept in buffer")
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

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------
DEVICE = args.device
if DEVICE == "cuda" and not torch.cuda.is_available():
    log.warning("CUDA not available, falling back to CPU")
    DEVICE = "cpu"
if args.dtype == "bf16" and DEVICE == "cuda":
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float16
log.info("Device=%s  dtype=%s", DEVICE, DTYPE)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_path = args.model_path
looks_local = model_path.startswith("/") or model_path.startswith("./") or model_path.startswith("../")
local_only = looks_local and os.path.isdir(model_path)
if looks_local and not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Local model path does not exist: {model_path}\n"
        "Mount the model or use a HuggingFace id: --model-path openbmb/MiniCPM-V-4_5"
    )

log.info("Loading model from %s …", model_path)
load_kwargs: dict = dict(trust_remote_code=True, attn_implementation="sdpa", torch_dtype=DTYPE)
if local_only:
    load_kwargs["local_files_only"] = True
model = AutoModel.from_pretrained(model_path, **load_kwargs)
model = model.to(device=DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
log.info("Model loaded.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 3
TIME_SCALE = 0.1
MAX_IMAGE_SIZE = 448 * 16

DEFAULT_CONFIG: Dict[str, Any] = {
    "enable_thinking": False,
    "sampling": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 100,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "system_prompt": "",
    "max_frames": args.max_frames,
    "video_fps": 3,
}

# ---------------------------------------------------------------------------
# Helpers — image / video
# ---------------------------------------------------------------------------

def resize_image(image: Image.Image) -> Image.Image:
    """Resize if too large, convert to RGB."""
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


def encode_video_file(video_path: str, choose_fps: int = 3, force_packing: Optional[int] = None):
    """Encode a video file → (frames, temporal_ids) using the 3D-Resampler pattern."""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = list(range(len(vr)))
    frame_idx = np.array(_uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)

    log.info("Video %s: duration=%.1fs  frames=%d  packing=%d", video_path, video_duration, len(frame_idx), packing_nums)

    raw_frames = vr.get_batch(frame_idx).asnumpy()
    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration + TIME_SCALE, TIME_SCALE)
    frame_ts_id = _map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    frames = [Image.fromarray(f.astype("uint8")).convert("RGB") for f in raw_frames]
    temporal_ids = _group_array(frame_ts_id, packing_nums)
    return frames, temporal_ids, {"duration": round(video_duration, 1), "frame_count": len(frames), "packing": packing_nums}


def build_temporal_ids_for_live_frames(n_frames: int, interval_ms: int = 200):
    """Generate temporal_ids for live webcam frames captured at a fixed interval."""
    if n_frames == 0:
        return []
    ts_seconds = np.arange(n_frames) * (interval_ms / 1000.0)
    scale = np.arange(0, ts_seconds[-1] + TIME_SCALE * 2, TIME_SCALE)
    ts_ids = _map_to_nearest_scale(ts_seconds, scale) / TIME_SCALE
    ts_ids = ts_ids.astype(np.int32)
    packing = min(max(1, n_frames // 10), MAX_NUM_PACKING)
    return _group_array(ts_ids, packing)


# ---------------------------------------------------------------------------
# Helpers — chat
# ---------------------------------------------------------------------------

def build_msgs(images: List[Image.Image], context: List[Dict], temporal_ids=None):
    """Build message list for model.chat(). Images/frames go in first user turn."""
    msgs = []
    for i, msg in enumerate(context):
        if msg["role"] == "user":
            text = msg["content"]
            if i == 0 and images:
                msgs.append({"role": "user", "content": images + [text]})
            else:
                msgs.append({"role": "user", "content": [text]})
        else:
            msgs.append({"role": "assistant", "content": [msg["content"]]})
    return msgs


def run_inference(
    images: List[Image.Image],
    context: List[Dict],
    config: Dict,
    temporal_ids=None,
    is_video: bool = False,
):
    """Run model.chat(stream=True). Yields text tokens."""
    if not images:
        yield "[error] No image or video context. Please upload media first."
        return

    msgs = build_msgs(images, context, temporal_ids)
    chat_kwargs: Dict[str, Any] = dict(
        msgs=msgs,
        tokenizer=tokenizer,
        enable_thinking=config.get("enable_thinking", False),
        stream=True,
        max_new_tokens=config.get("max_new_tokens", 2048),
        system_prompt=config.get("system_prompt", ""),
    )
    if config.get("sampling", True):
        chat_kwargs.update(
            sampling=True,
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 100),
            temperature=config.get("temperature", 0.7),
            repetition_penalty=config.get("repetition_penalty", 1.05),
        )
    else:
        chat_kwargs.update(sampling=False, num_beams=3, repetition_penalty=1.2)

    if is_video and temporal_ids:
        chat_kwargs["temporal_ids"] = temporal_ids
        chat_kwargs["use_image_id"] = False
        chat_kwargs["max_slice_nums"] = 1
    elif is_video:
        n = len(images)
        chat_kwargs["max_inp_length"] = 4352
        chat_kwargs["use_image_id"] = False
        chat_kwargs["max_slice_nums"] = 1 if n > 16 else 2

    try:
        gen = model.chat(**chat_kwargs)
        for token in gen:
            yield token
    except Exception as e:
        log.exception("Inference failed: %s", e)
        yield f"[error] Inference failed: {e}"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MiniCPM-V 4.5", version="1.0.0")
_executor = ThreadPoolExecutor(max_workers=2)

SRC_DIR = Path(__file__).resolve().parent


@app.get("/")
async def index():
    return FileResponse(SRC_DIR / "live.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "MiniCPM-V-4.5", "device": DEVICE}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

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

    await send("session_ready", {"max_frames": args.max_frames, "config": session["config"]})

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

            # ── upload_images (Tab 1) ─────────────────────────────
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
                    session["images"] = imgs
                    session["video_frames"] = []
                    session["video_temporal_ids"] = None
                    session["context"] = []
                    session["mode"] = "photo"
                    await send("upload_done", {"type": "image", "count": len(imgs)})
                except Exception as e:
                    log.exception("upload_images failed")
                    await send("error", {"message": str(e)})
                continue

            # ── upload_video (Tab 2) ──────────────────────────────
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
                    session["video_frames"] = frames
                    session["video_temporal_ids"] = temporal_ids
                    session["video_info"] = info
                    session["images"] = []
                    session["context"] = []
                    session["mode"] = "video"
                    await send("upload_done", {"type": "video", **info})
                except Exception as e:
                    log.exception("upload_video failed")
                    await send("error", {"message": str(e)})
                continue

            # ── frame (Tab 3 — continuous webcam) ─────────────────
            if action == "frame":
                b64_data = msg.get("data")
                if not b64_data:
                    continue
                try:
                    raw_bytes = base64.b64decode(b64_data)
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    img = resize_image(img)
                    with session["lock"]:
                        session["frame_buffer"].append(img)
                        session["frame_timestamps"].append(time.time())
                except Exception:
                    pass
                continue

            # ── chat ──────────────────────────────────────────────
            if action == "chat":
                text = (msg.get("text") or "").strip()
                if not text:
                    await send("error", {"message": "Empty message"})
                    continue

                mode = msg.get("mode", session["mode"])
                session["mode"] = mode

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
                        await send("error", {"message": "No webcam frames received yet"})
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

                if not session["context"]:
                    text = text
                session["context"].append({"role": "user", "content": text})

                await send("chat_started", {"frame_count": len(media), "mode": mode})

                loop = asyncio.get_event_loop()
                config = dict(session["config"])
                ctx_snapshot = list(session["context"])

                full_text = ""
                try:
                    gen = await loop.run_in_executor(
                        _executor,
                        lambda: list(
                            _stream_wrapper(media, ctx_snapshot, config, temporal_ids, is_video)
                        ),
                    )
                    for token in gen:
                        full_text += token
                        await send("chat_token", {"text": token})
                except WebSocketDisconnect:
                    log.info("Client disconnected during chat streaming")
                    session["context"].pop()
                    raise
                except Exception as e:
                    log.exception("chat streaming error")
                    await send("error", {"message": str(e)})
                    session["context"].pop()
                    continue

                full_text = _clean_response(full_text)
                session["context"].append({"role": "assistant", "content": full_text})
                await send("chat_done", {"text": full_text})
                continue

            # ── set_config ────────────────────────────────────────
            if action == "set_config":
                new_cfg = msg.get("config", {})
                for k, v in new_cfg.items():
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

            # ── regenerate ────────────────────────────────────────
            if action == "regenerate":
                if len(session["context"]) < 2:
                    await send("error", {"message": "Nothing to regenerate"})
                    continue
                session["context"].pop()  # remove last assistant
                last_user = session["context"].pop()  # remove last user
                fake_msg = {"action": "chat", "text": last_user["content"], "mode": session["mode"]}
                await websocket.send_json({"_inject": fake_msg})
                # Re-process as a chat action
                raw = json.dumps(fake_msg)
                msg = fake_msg
                action = "chat"
                # Fall through won't work cleanly — just re-send
                text = last_user["content"]
                mode = session["mode"]

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

                if not media:
                    await send("error", {"message": "No media available"})
                    continue

                session["context"].append({"role": "user", "content": text})
                await send("chat_started", {"frame_count": len(media), "mode": mode})

                loop = asyncio.get_event_loop()
                config = dict(session["config"])
                ctx_snapshot = list(session["context"])
                full_text = ""
                try:
                    gen = await loop.run_in_executor(
                        _executor,
                        lambda: list(_stream_wrapper(media, ctx_snapshot, config, temporal_ids, is_video)),
                    )
                    for token in gen:
                        full_text += token
                        await send("chat_token", {"text": token})
                except WebSocketDisconnect:
                    log.info("Client disconnected during regenerate streaming")
                    session["context"].pop()
                    raise
                except Exception as e:
                    log.exception("regenerate error")
                    await send("error", {"message": str(e)})
                    session["context"].pop()
                    continue
                full_text = _clean_response(full_text)
                session["context"].append({"role": "assistant", "content": full_text})
                await send("chat_done", {"text": full_text})
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


def _stream_wrapper(images, context, config, temporal_ids, is_video):
    """Wrapper to collect streaming tokens (runs in thread pool)."""
    tokens = []
    for token in run_inference(images, context, config, temporal_ids, is_video):
        tokens.append(token)
    return tokens


def _clean_response(text: str) -> str:
    text = re.sub(r"(<box>.*?</box>)", "", text)
    text = text.replace("<ref>", "").replace("</ref>", "")
    text = text.replace("<box>", "").replace("</box>", "")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
