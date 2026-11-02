#!/usr/bin/env python3
"""
TDR · Discord controller + TinyML print-failure monitor
=======================================================

Ender 3 (SKR Mini E3 V3) · Klipper · Moonraker · Crowsnest · Raspberry Pi 4B (4 GB)

What it does
------------
* Runs a MobileNetV2 ``.tflite`` classifier on frames from the nozzle camera while
  the printer is printing (idle -> sleeps, ~0 % CPU).
* Debounces detections (N consecutive frames above a confidence threshold).
* Supervised mode  -> DMs you the evidence frame with Stop / Pause / Ignore buttons
                      (optional auto-pause if you don't answer in time).
  Autonomous mode -> pauses/cancels via Moonraker immediately, then DMs you.
* ``/check`` on-demand inference, ``/printer`` dashboard, completion / Klipper error DMs.

RAM / CPU budget rules (the Pi also runs Klipper, Moonraker, Mainsail, Crowsnest)
-------------------------------------------------------------------------------
* LiteRT / tflite-runtime interpreter (no full TensorFlow) with ``num_threads=2``.
* Inference runs in ONE dedicated worker thread -> never blocks the event loop and
  never runs two inferences in parallel.
* Frames are pulled from the Crowsnest HTTP snapshot endpoint (no camera contention);
  Picamera2 is only a fallback and is released whenever the printer goes idle.
* One shared ``aiohttp.ClientSession``; no blocking ``requests``.
* Images only ever live in ``io.BytesIO`` / numpy buffers -> zero temp files on the SD card.
* discord.py with no member cache and no message cache.

Model compatibility
-------------------
The bot unifies all models into a reliable 2-class output ``["Correcte", "Error"]``:
* 3+ outputs -> ``Correcte = P(class 0)``, ``Error = 1 - P(class 0)`` (combines all failure modes)
* 1 output   -> binary sigmoid ``P(Error)`` mapped to ``["Correcte", "Error"]``.
* Multi-scale inference evaluates both the Nozzle ROI crop and the Full Frame so both
  localized nozzle failures (clogs/under-extrusion/air-printing) and wide spaghetti are caught.
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import math
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

import aiohttp
import cv2
import discord
import numpy as np
from discord import app_commands
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CLASSES = ["Correcte", "Error"]
OK_CLASS = "Correcte"
ERROR_CLASS = "Error"
DEFAULT_ROI_BOX = (0.25, 0.32, 0.86, 0.96)  # (x1, y1, x2, y2) focused on nozzle + print zone, excluding BLTouch & V-wheel
DEFAULT_ROI_STR = "0.25,0.32,0.86,0.96"

MODEL_CANDIDATES = (
    "printer_failure_detector.tflite",
    "printWatcher_efficientnet_v2s.tflite",
    "model_efficientnet_v2s.tflite",
    "model_errors_cambridge_definitiu.tflite",
    "model_errors_cambridge.tflite",
    "model_errors_3d.tflite",
)

CLASS_EMOJI = {"Correcte": "✅", "Error": "🚨"}
STATE_EMOJI = {
    "printing": "🟢", "paused": "⏸️", "standby": "💤", "complete": "🏁", "cancelled": "🚫",
    "error": "🔴", "offline": "⚫", "startup": "🔄", "shutdown": "🔴", "disconnected": "⚫",
}
STATE_COLOR = {
    "printing": discord.Color.from_rgb(64, 156, 255),
    "paused": discord.Color.from_rgb(255, 170, 0),
    "complete": discord.Color.from_rgb(46, 204, 113),
    "error": discord.Color.from_rgb(231, 76, 60),
    "cancelled": discord.Color.from_rgb(149, 165, 166),
}
COLOR_OK = discord.Color.from_rgb(46, 204, 113)
COLOR_WARN = discord.Color.from_rgb(255, 170, 0)
COLOR_BAD = discord.Color.from_rgb(231, 76, 60)
COLOR_INFO = discord.Color.from_rgb(114, 137, 218)
COLOR_IDLE = discord.Color.from_rgb(99, 110, 114)

HIGH_TEMP_C = 240.0  # PETG / ABS territory on an Ender 3 hotend

cv2.setNumThreads(1)  # OpenCV must not compete with Klipper / TFLite for cores
log = logging.getLogger("tdr.bot")


# --------------------------------------------------------------------------------------
# Configuration (.env)
# --------------------------------------------------------------------------------------
def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or not v.strip() else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        log.warning("Invalid float for %s, using %s", name, default)
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        log.warning("Invalid int for %s, using %s", name, default)
        return default


def _resolve_model_path(raw: str) -> Optional[Path]:
    """MODEL_PATH from .env first, then the known file names next to the script / one level up."""
    candidates: list[Path] = []
    if raw:
        p = Path(raw).expanduser()
        candidates.append(p if p.is_absolute() else BASE_DIR / p)
    for d in (BASE_DIR, BASE_DIR.parent):
        candidates.extend(d / name for name in MODEL_CANDIDATES)
    return next((c for c in candidates if c.is_file()), None)


@dataclass(frozen=True)
class Config:
    token: str
    user_id: int
    extra_user_ids: frozenset[int]
    alert_channel_id: int
    model_path: Optional[Path]
    num_threads: int
    confidence_threshold: float
    consecutive_frames: int
    check_interval: float
    idle_poll_interval: float
    watch_interval: float
    supervised_timeout: int
    autonomous_action: str
    default_mode: str
    ai_enabled_default: bool
    snooze_seconds: int
    save_false_positives: bool
    max_false_positives: int
    force_ai_when_idle: bool
    moonraker_url: str
    moonraker_api_key: str
    snapshot_url: str
    snapshot_timeout: float
    snapshot_max_width: int
    flip_snapshot_180: bool
    flip_camera_180: bool
    lens_position: float
    picam_width: int
    picam_height: int
    jpeg_quality: int
    notify_startup: bool
    notify_complete: bool
    notify_cancelled: bool
    notify_errors: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        token = _env_str("DISCORD_BOT_TOKEN")
        user_id = _env_int("DISCORD_USER_ID", 0)
        if not token or not user_id:
            raise SystemExit("❌ DISCORD_BOT_TOKEN and DISCORD_USER_ID must be set in AI/.env (run ./setup.sh)")
        extra = frozenset(int(x) for x in _env_str("EXTRA_ALLOWED_USER_IDS").replace(" ", "").split(",") if x.isdigit())
        mode = _env_str("AI_MODE", "supervised").lower()
        action = _env_str("AUTONOMOUS_ACTION", "pause").lower()
        return cls(
            token=token,
            user_id=user_id,
            extra_user_ids=extra,
            alert_channel_id=_env_int("DISCORD_ALERT_CHANNEL_ID", 0),
            model_path=_resolve_model_path(_env_str("MODEL_PATH")),
            num_threads=max(1, _env_int("TFLITE_THREADS", 2)),
            confidence_threshold=min(0.999, max(0.05, _env_float("CONFIDENCE_THRESHOLD", 0.60))),
            consecutive_frames=max(1, _env_int("CONSECUTIVE_FRAMES", 4)),
            check_interval=max(0.5, _env_float("CHECK_INTERVAL", 2.5)),
            idle_poll_interval=max(2.0, _env_float("IDLE_POLL_INTERVAL", 10.0)),
            watch_interval=max(2.0, _env_float("WATCH_INTERVAL", 5.0)),
            supervised_timeout=max(0, _env_int("SUPERVISED_TIMEOUT", 90)),
            autonomous_action=action if action in ("pause", "cancel") else "pause",
            default_mode=mode if mode in ("supervised", "autonomous") else "supervised",
            ai_enabled_default=_env_bool("AI_ENABLED", True),
            snooze_seconds=max(0, _env_int("SNOOZE_SECONDS", 60)),
            save_false_positives=_env_bool("SAVE_FALSE_POSITIVES", True),
            max_false_positives=max(1, _env_int("MAX_FALSE_POSITIVES", 50)),
            force_ai_when_idle=_env_bool("FORCE_AI_WHEN_IDLE", False),
            moonraker_url=_env_str("MOONRAKER_URL", "http://localhost:7125").rstrip("/"),
            moonraker_api_key=_env_str("MOONRAKER_API_KEY"),
            snapshot_url=_env_str("CAMERA_SNAPSHOT_URL", "http://127.0.0.1:8080/?action=snapshot"),
            snapshot_timeout=max(0.5, _env_float("SNAPSHOT_TIMEOUT", 3.0)),
            snapshot_max_width=max(320, _env_int("SNAPSHOT_MAX_WIDTH", 1024)),
            flip_snapshot_180=_env_bool("FLIP_SNAPSHOT_180", False),
            flip_camera_180=_env_bool("FLIP_CAMERA_180", True),
            lens_position=_env_float("LENS_POSITION", 10.0),
            picam_width=_env_int("PICAM_WIDTH", 640),
            picam_height=_env_int("PICAM_HEIGHT", 480),
            jpeg_quality=min(95, max(40, _env_int("JPEG_QUALITY", 85))),
            notify_startup=_env_bool("NOTIFY_STARTUP", True),
            notify_complete=_env_bool("NOTIFY_PRINT_COMPLETE", True),
            notify_cancelled=_env_bool("NOTIFY_PRINT_CANCELLED", True),
            notify_errors=_env_bool("NOTIFY_KLIPPER_ERRORS", True),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        )


# --------------------------------------------------------------------------------------
# Persistent runtime state (tiny JSON, written only when a setting changes)
# --------------------------------------------------------------------------------------
STATE_FILE = BASE_DIR / "bot_state.json"


def parse_roi_box(raw: str) -> tuple[float, float, float, float]:
    """Parse 'x1,y1,x2,y2' normalized coordinates in [0..1]; fall back to DEFAULT_ROI_BOX."""
    try:
        parts = [float(v.strip()) for v in raw.split(",")]
        if len(parts) == 4:
            x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in parts)
            if x2 - x1 >= 0.15 and y2 - y1 >= 0.15:
                return (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
    except (ValueError, AttributeError):
        pass
    return DEFAULT_ROI_BOX


def crop_roi(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Extract normalized (x1, y1, x2, y2) nozzle ROI sub-frame."""
    h, w = rgb.shape[:2]
    x1, y1 = int(round(box[0] * w)), int(round(box[1] * h))
    x2, y2 = int(round(box[2] * w)), int(round(box[3] * h))
    if x2 - x1 >= 32 and y2 - y1 >= 32:
        return rgb[y1:y2, x1:x2]
    return rgb


def detect_nozzle_air_gap(rgb: np.ndarray, min_gap_ratio: float = 0.016) -> float:
    """Lighting-invariant Skyline Air-Gap Clog Detector for a carriage-mounted nozzle camera.

    When a nozzle clogs (or filament runs out / strips), Klipper keeps stepping Z upwards while
    extrusion stops: an open horizontal band of smooth out-of-focus background bed appears directly
    between the fixed dark nozzle cone tip and the top surface of the stopped 3D print below.
    Returns P(Error) in [0.0 .. 0.95] (0.0 when touching part, Z-hopping <1.5mm, travelling
    between towers, printing first-layer skirts, or hovering over an empty bed).
    """
    fh, fw = rgb.shape[:2]
    if fh < 120 or fw < 160:
        return 0.0
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

    # 1. Locate nozzle cone tip bottom row (ny); 15th percentile ignores thin <4px ooze threads.
    nx0, nx1 = int(0.53 * fw), int(0.61 * fw)
    ys0, ys1 = int(0.44 * fh), int(0.59 * fh)
    cone_patch = gray[ys0:ys1, nx0:nx1]
    cone_p15 = np.percentile(cone_patch, 15, axis=1)
    nozzle_dark_ref = float(np.min(cone_p15[:max(3, int(0.05 * fh))]))

    left_open_bed = float(np.median(gray[int(0.50 * fh):int(0.60 * fh), int(0.06 * fw):int(0.20 * fw)]))
    if left_open_bed <= nozzle_dark_ref + 25.0:
        return 0.0  # dark room / obstructed lens
    cone_thresh = nozzle_dark_ref + 0.36 * (left_open_bed - nozzle_dark_ref)

    ny = ys0
    for i, val in enumerate(cone_p15):
        if val < cone_thresh:
            ny = ys0 + i
        elif i > int(0.03 * fh):
            break

    if ny >= ys1 - 3:
        return 0.0  # dark printed part is directly touching the nozzle tip

    y0 = min(fh - 30, ny + 3)
    y1 = int(0.90 * fh)
    if y1 - y0 < int(0.10 * fh):
        return 0.0

    zx0, zx1 = int(0.42 * fw), int(0.70 * fw)
    zone = gray[y0:y1, zx0:zx1]
    row_med = np.median(zone, axis=1)
    dy_med = np.mean(np.abs(np.diff(zone, axis=0, prepend=zone[:1])), axis=1)

    top_n = max(3, int(0.010 * fh))
    init_lum = float(np.median(row_med[:top_n]))
    init_dy = float(np.mean(dy_med[:top_n]))

    # 2. Reject travel moves across gaps between active towers reaching the nozzle plane.
    th = max(6, int(0.018 * fh))
    flank_l = gray[y0:y0 + th, int(0.24 * fw):int(0.46 * fw)]
    flank_r = gray[y0:y0 + th, int(0.66 * fw):int(0.77 * fw)]
    dy_l = float(np.abs(np.diff(flank_l, axis=0)).mean())
    dy_r = float(np.abs(np.diff(flank_r, axis=0)).mean())
    dark_l = float((flank_l < cone_thresh).mean())
    dark_r = float((flank_r < cone_thresh).mean())
    if dy_l > 9.5 or dy_r > 9.5 or dark_l > 0.35 or dark_r > 0.35:
        return 0.0

    # Air gap right below nozzle must look like shaded out-of-focus background bed.
    if not (cone_thresh + 6.0 <= init_lum <= left_open_bed * 1.08 and init_dy < 8.5):
        return 0.0

    # 3. Walk downward through smooth air gap until hitting a substantial 3D-printed body.
    min_gap_px = max(8, int(min_gap_ratio * fh))
    max_gap_px = int(0.24 * fh)
    body_win = max(18, int(0.075 * fh))
    jump_thr = max(20.0, 0.24 * init_lum)

    gap_px = 0
    running_lum = init_lum
    part_hit = False

    for i in range(min(len(row_med), max_gap_px + 1)):
        lum_jump = abs(float(row_med[i]) - running_lum)
        sharp_jump = float(dy_med[i])
        if lum_jump > jump_thr or sharp_jump > 9.0:
            ahead = slice(i, min(len(row_med), i + body_win))
            if ahead.stop - ahead.start < max(10, body_win // 2):
                break
            row_is_part = (np.abs(row_med[ahead] - running_lum) > jump_thr) | (dy_med[ahead] > 8.5)
            # >=60% of the 42px vertical window must be printed body (ignores thin 1-layer skirt lines)
            if float(row_is_part.mean()) >= 0.60:
                if i < min_gap_px:
                    return 0.0  # part touches nozzle within normal extrusion/Z-hop tolerance
                part_hit = True
                break
        gap_px += 1
        running_lum = 0.94 * running_lum + 0.06 * float(row_med[i])

    if not part_hit:
        return 0.0  # open empty bed with no detached print below

    ahead = slice(gap_px, min(len(row_med), gap_px + body_win))
    part_contrast = float(np.median(np.abs(row_med[ahead] - init_lum)))
    gap_frac = gap_px / float(fh)
    return float(min(0.95, 0.74 + 2.2 * min(0.08, max(0.0, gap_frac - 0.015)) + 0.0012 * min(80.0, part_contrast)))


@dataclass
class AISettings:
    enabled: bool
    mode: str                 # "supervised" | "autonomous"
    threshold: float
    frames: int
    supervised_timeout: int   # seconds, 0 = never auto-pause
    active_model: str = ""    # saved .tflite filename or "ensemble"
    roi_mode: str = "roi"     # "roi" (nozzle ROI + air-gap) | "multi" (ROI + full + air-gap) | "full"
    roi_box: str = DEFAULT_ROI_STR
    schema_version: int = 3

    @classmethod
    def load(cls, cfg: Config) -> "AISettings":
        env_roi_mode = _env_str("AI_ROI_MODE", "roi").lower()
        if env_roi_mode not in ("multi", "roi", "full"):
            env_roi_mode = "roi"
        env_roi_box = ",".join(f"{v:.2f}" for v in parse_roi_box(_env_str("AI_ROI_BOX", DEFAULT_ROI_STR)))
        s = cls(cfg.ai_enabled_default, cfg.default_mode, cfg.confidence_threshold,
                cfg.consecutive_frames, cfg.supervised_timeout, "ensemble", env_roi_mode, env_roi_box, 3)
        migrated = False
        try:
            data = json.loads(STATE_FILE.read_text())
            prev_ver = int(data.get("schema_version", 1))
            for k in asdict(s):
                if k in data:
                    setattr(s, k, type(getattr(s, k))(data[k]))
            if s.roi_mode not in ("multi", "roi", "full"):
                s.roi_mode = "roi"
            s.roi_box = ",".join(f"{v:.2f}" for v in parse_roi_box(s.roi_box))
            if prev_ver < 2 and s.threshold >= 0.78:
                s.threshold = 0.60
                migrated = True
            # Auto-migrate v1/v2 state files to v3 (motion-robust Dual-Brain Ensemble + Nozzle ROI + 4/6 debounce)
            if prev_ver < 3:
                log.info("Migrating bot_state.json v%d -> v3 (Dual-Brain Ensemble + BLTouch-safe ROI + 4/6 window)",
                         prev_ver)
                if s.active_model in ("", "printWatcher_efficientnet_v2s.tflite", "model_efficientnet_v2s.tflite"):
                    s.active_model = "ensemble"
                s.roi_mode = "roi"
                s.roi_box = DEFAULT_ROI_STR
                if s.frames < 4:
                    s.frames = 4
                s.schema_version = 3
                migrated = True
        except FileNotFoundError:
            pass
        except Exception as e:  # corrupted file -> defaults
            log.warning("Ignoring unreadable %s: %s", STATE_FILE.name, e)
        if migrated:
            s.save()
        return s

    def save(self) -> None:
        try:
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2))
            tmp.replace(STATE_FILE)
        except OSError as e:
            log.warning("Could not persist settings: %s", e)


# --------------------------------------------------------------------------------------
# TFLite classifier
# --------------------------------------------------------------------------------------
def _import_interpreter():
    """Lightest runtime first: LiteRT -> tflite-runtime -> full TensorFlow (last resort)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai-edge-litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite-runtime"
    except ImportError:
        pass
    import tensorflow as tf  # heavy (~500 MB RAM) – only if nothing else is installed
    return tf.lite.Interpreter, "tensorflow"


@dataclass
class Prediction:
    probs: dict[str, float]   # {"Correcte": p_ok, "Error": p_err}
    ms: float                 # inference latency
    source: str = ""          # camera source used for the frame
    view: str = ""            # "nozzle-roi" | "full-frame"

    @property
    def label(self) -> str:
        return max(self.probs, key=self.probs.get)

    @property
    def confidence(self) -> float:
        return self.probs[self.label]

    @property
    def ok_conf(self) -> float:
        return float(self.probs.get(OK_CLASS, 0.0))

    @property
    def failure_label(self) -> str:
        return ERROR_CLASS

    @property
    def failure_conf(self) -> float:
        # Total probability of failure = P(Error) = 1 - P(Correcte)
        if ERROR_CLASS in self.probs:
            return float(self.probs[ERROR_CLASS])
        return float(max(0.0, 1.0 - self.ok_conf))

    def is_alarm(self, threshold: float) -> bool:
        return self.failure_conf >= threshold


class TFLiteClassifier:
    """Thin, thread-safe wrapper. Unifies 1-output and multi-class models into ['Correcte', 'Error']."""

    def __init__(self, path: Path, num_threads: int = 2):
        Interpreter, self.backend = _import_interpreter()
        self.path = path
        self.num_threads = num_threads
        self.interp = Interpreter(model_path=str(path), num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.h, self.w, _ = (int(v) for v in self.inp["shape"])
        self.n_out = int(np.prod(self.out["shape"]))
        self.classes = CLASSES
        if self.n_out == 1:
            self.kind = "binary-sigmoid"
        else:
            self.kind = f"{self.n_out}-out->binary(Correcte/Error)"
        self._lock = threading.Lock()
        self.last_ms: float = 0.0
        self.size_mb: float = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        self.predict(np.zeros((self.h, self.w, 3), np.uint8))  # warm-up: allocates XNNPACK buffers once
        log.info("Model %s loaded via %s (%s, input %s %s, %d threads)", path.name, self.backend,
                 self.kind, list(self.inp["shape"]), np.dtype(self.inp["dtype"]).name, num_threads)

    @property
    def resolution_str(self) -> str:
        return f"{self.w}x{self.h}"

    @property
    def is_efficientnet(self) -> bool:
        nl = self.path.name.lower()
        return "efficientnet" in nl or "printwatcher" in nl or self.w >= 280

    @property
    def display_name(self) -> str:
        nl = self.path.name.lower()
        if "printwatcher" in nl:
            return "PrintWatcher (EfficientNetV2S)"
        if "efficientnet" in nl:
            return "EfficientNetV2S"
        if "cambridge" in nl:
            return "MobileNetV2 (Cambridge)"
        if "printer_failure_detector" in nl:
            return "MobileNetV2 (Binary)"
        return self.path.stem

    def close(self) -> None:
        with self._lock:
            self.interp = None

    def _prepare(self, rgb: np.ndarray) -> np.ndarray:
        # Same preprocessing as Keras image_dataset_from_directory training: RGB, bilinear resize,
        # raw 0-255 pixels (preserves thin filament strands). Rescaling is baked into the model graph.
        if rgb.shape[:2] != (self.h, self.w):
            rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            return rgb.astype(np.float32)[None]
        scale, zp = self.inp["quantization"]
        if not scale:  # quantized tensor without params -> raw pixels
            return rgb.astype(dtype)[None]
        info = np.iinfo(dtype)
        q = np.clip(np.round(rgb.astype(np.float32) / scale + zp), info.min, info.max)
        return q.astype(dtype)[None]

    def predict(self, rgb: np.ndarray) -> Prediction:
        x = self._prepare(rgb)
        t0 = time.perf_counter()
        with self._lock:
            if self.interp is None:
                raise RuntimeError(f"Model {self.path.name} is closed")
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"]).copy()
        ms = (time.perf_counter() - t0) * 1000
        self.last_ms = ms
        if self.out["dtype"] in (np.uint8, np.int8):
            scale, zp = self.out["quantization"]
            y = (y.astype(np.float32) - zp) * (scale or 1.0)
        y = y.astype(np.float32).reshape(-1)
        if self.n_out == 1:
            p_err = float(y[0])
            if not 0.0 <= p_err <= 1.0:  # logits -> sigmoid
                p_err = float(1.0 / (1.0 + np.exp(-p_err)))
            p_err = float(np.clip(p_err, 0.0, 1.0))
            p_ok = 1.0 - p_err
        else:
            if y.min() < 0 or abs(float(y.sum()) - 1.0) > 0.02:  # logits -> softmax
                e = np.exp(y - y.max())
                y = e / e.sum()
            total = float(y.sum()) or 1.0
            p_ok = float(np.clip(y[0] / total, 0.0, 1.0))
            # Collapse all failure classes (e.g. Espagueti + Obstruccio) into unified Error probability
            p_err = float(np.clip(1.0 - p_ok, 0.0, 1.0))
        return Prediction({OK_CLASS: p_ok, ERROR_CLASS: p_err}, ms)


class ModelManager:
    """Manages discovery, dynamic resolution adaptation, thread-safe hot-swapping,
    and Ensemble Mode across all .tflite models in AI/ and project directory."""

    def __init__(self, bot: "TDRBot"):
        self.bot = bot
        self.active_key: str = ""       # filename or "ensemble"
        self.classifier: Optional[TFLiteClassifier] = None
        self.ensemble_classifiers: list[TFLiteClassifier] = []
        self._lock = asyncio.Lock()
        self.model_error: str = ""
        self.last_ms: float = 0.0

    def discover_models(self) -> list[Path]:
        """Auto-discover all *.tflite files in AI/ and parent dir, prioritized."""
        found: dict[str, Path] = {}
        # 1. Candidate names in order (EfficientNetV2S prioritized at the top)
        for name in MODEL_CANDIDATES:
            for d in (BASE_DIR, BASE_DIR.parent):
                p = d / name
                if p.is_file() and p.name not in found:
                    found[p.name] = p
        # 2. Configured MODEL_PATH from .env
        raw = _env_str("MODEL_PATH")
        if raw:
            p = Path(raw).expanduser()
            target = p if p.is_absolute() else BASE_DIR / p
            if target.is_file() and target.name not in found:
                found[target.name] = target
        # 3. Any other *.tflite in AI/ and parent dir
        for d in (BASE_DIR, BASE_DIR.parent):
            if d.is_dir():
                for p in sorted(d.glob("*.tflite")):
                    if p.name not in found:
                        found[p.name] = p
        return list(found.values())

    def is_ready(self) -> bool:
        if self.active_key == "ensemble":
            return len(self.ensemble_classifiers) >= 2
        return self.classifier is not None

    async def initialize(self) -> None:
        """Called during setup_hook to load initial model."""
        models = self.discover_models()
        if not models:
            self.model_error = "No .tflite models found in AI/ or project directory"
            log.error(self.model_error)
            return

        saved = self.bot.settings.active_model
        if (saved == "ensemble" or not saved) and len(models) >= 2:
            target = "ensemble"
        elif saved and any(m.name == saved for m in models):
            target = saved
        else:
            mob = next((m for m in models if "printer_failure_detector" in m.name.lower()), None)
            target = mob.name if mob else models[0].name

        await self.switch_model(target)

    async def switch_model(self, target_key: str) -> tuple[bool, str]:
        """Thread-safe hot-swap of active model or ensemble mode."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_switch, target_key)

    def _sync_switch(self, target_key: str) -> tuple[bool, str]:
        models = self.discover_models()
        model_map = {m.name: m for m in models}

        # Release previous interpreter(s) and free RAM
        if self.classifier:
            self.classifier.close()
            self.classifier = None
        for c in self.ensemble_classifiers:
            c.close()
        self.ensemble_classifiers.clear()
        gc.collect()

        threads = self.bot.cfg.num_threads
        if target_key == "ensemble":
            if len(models) < 2:
                return False, "Ensemble requires at least 2 .tflite models in AI/"
            m1 = next((m for m in models if "efficientnet" in m.name.lower()), models[0])
            m2 = next((m for m in models if m.name != m1.name), models[1])
            try:
                c1 = TFLiteClassifier(m1, num_threads=threads)
                c2 = TFLiteClassifier(m2, num_threads=threads)
                self.ensemble_classifiers = [c1, c2]
                self.active_key = "ensemble"
                self.model_error = ""
                self.bot.settings.active_model = "ensemble"
                self.bot.settings.save()
                log.info("🧠 Ensemble Mode activated: %s + %s", m1.name, m2.name)
                return True, f"Ensemble Mode activated ({m1.name} + {m2.name})"
            except Exception as e:
                self.model_error = f"Ensemble load failed: {e}"
                log.exception(self.model_error)
                return False, self.model_error
        else:
            if target_key not in model_map:
                return False, f"Model '{target_key}' not found"
            try:
                path = model_map[target_key]
                self.classifier = TFLiteClassifier(path, num_threads=threads)
                self.active_key = target_key
                self.model_error = ""
                self.bot.settings.active_model = target_key
                self.bot.settings.save()
                log.info("Active model switched to: %s", target_key)
                return True, f"Active model switched to {target_key}"
            except Exception as e:
                self.model_error = f"Model load failed: {e}"
                log.exception(self.model_error)
                return False, self.model_error

    def _predict_single_view(self, view_rgb: np.ndarray) -> Prediction:
        """Run active model (or weighted ensemble) on a single RGB view."""
        if self.active_key == "ensemble":
            if len(self.ensemble_classifiers) < 2:
                raise RuntimeError("Ensemble classifiers not loaded")
            c1, c2 = self.ensemble_classifiers[0], self.ensemble_classifiers[1]
            p1 = c1.predict(view_rgb)
            p2 = c2.predict(view_rgb)
            # Favor motion-blur/LED-augmented MobileNetV2 (0.85) + EfficientNetV2S (0.15)
            # to eliminate false alarms during rapid Y-bed motion under red BLTouch lighting.
            if c1.is_efficientnet and not c2.is_efficientnet:
                w1, w2 = 0.15, 0.85
            elif c2.is_efficientnet and not c1.is_efficientnet:
                w1, w2 = 0.85, 0.15
            else:
                w1, w2 = 0.5, 0.5
            p_err = float(np.clip(w1 * p1.failure_conf + w2 * p2.failure_conf, 0.0, 1.0))
            p_ok = float(1.0 - p_err)
            return Prediction({OK_CLASS: p_ok, ERROR_CLASS: p_err}, p1.ms + p2.ms)
        if self.classifier is None:
            raise RuntimeError(self.model_error or "No model loaded")
        return self.classifier.predict(view_rgb)

    def predict(self, frame: np.ndarray) -> Prediction:
        """Hybrid Multi-Scale CNN + Skyline Air-Gap Clog prediction pass (worker thread).

        * 'multi' (default): evaluates Nozzle ROI CNN (under-extrusion / broken infill),
          Full-Frame CNN (spaghetti / detached prints), and Skyline Air-Gap Clog detection
          (full clogs / filament runout / air-printing), keeping the strongest Error signal.
        * 'roi': evaluates Nozzle ROI CNN + Skyline Air-Gap Clog detection.
        * 'full': evaluates Full-Frame CNN only.
        """
        roi_mode = self.bot.settings.roi_mode
        box = parse_roi_box(self.bot.settings.roi_box)

        p_state = getattr(self.bot, "printer_state", "unknown")
        z_mm = getattr(self.bot, "printer_z", None)
        allow_air_gap = not (p_state == "printing" and z_mm is not None and 0.0 <= z_mm < 1.2)

        if roi_mode == "full":
            pred = self._predict_single_view(frame)
            pred.view = "full-frame"
        elif roi_mode == "roi":
            p_roi = self._predict_single_view(crop_roi(frame, box))
            t_gap0 = time.perf_counter()
            p_gap = detect_nozzle_air_gap(frame) if allow_air_gap else 0.0
            gap_ms = (time.perf_counter() - t_gap0) * 1000
            if p_gap > p_roi.failure_conf:
                p_err, view_used = p_gap, "air-gap-clog"
            else:
                p_err, view_used = p_roi.failure_conf, "nozzle-roi"
            pred = Prediction({OK_CLASS: 1.0 - p_err, ERROR_CLASS: p_err}, p_roi.ms + gap_ms, view=view_used)
        else:
            p_roi = self._predict_single_view(crop_roi(frame, box))
            p_full = self._predict_single_view(frame)
            t_gap0 = time.perf_counter()
            p_gap = detect_nozzle_air_gap(frame) if allow_air_gap else 0.0
            gap_ms = (time.perf_counter() - t_gap0) * 1000
            total_ms = p_roi.ms + p_full.ms + gap_ms
            candidates = [
                (p_roi.failure_conf, "nozzle-roi"),
                (p_full.failure_conf, "full-frame"),
                (p_gap, "air-gap-clog"),
            ]
            p_err, view_used = max(candidates, key=lambda item: item[0])
            pred = Prediction({OK_CLASS: 1.0 - p_err, ERROR_CLASS: p_err}, total_ms, view=view_used)

        self.last_ms = pred.ms
        return pred

    async def close(self) -> None:
        async with self._lock:
            if self.classifier:
                self.classifier.close()
                self.classifier = None
            for c in self.ensemble_classifiers:
                c.close()
            self.ensemble_classifiers.clear()
            gc.collect()

    @property
    def display_name(self) -> str:
        if self.active_key == "ensemble":
            names = [c.path.stem for c in self.ensemble_classifiers]
            return f"🧠 Ensemble ({' + '.join(names)})"
        if self.classifier:
            return self.classifier.display_name
        return "None"

    @property
    def summary_string(self) -> str:
        if self.active_key == "ensemble":
            res = "+".join(c.resolution_str for c in self.ensemble_classifiers)
            return f"🧠 Ensemble ({res})"
        if self.classifier:
            return f"{self.classifier.display_name} ({self.classifier.resolution_str})"
        return "No Model"

    @property
    def resolution_str(self) -> str:
        if self.active_key == "ensemble":
            return "/".join(c.resolution_str for c in self.ensemble_classifiers)
        if self.classifier:
            return self.classifier.resolution_str
        return "—"

    @property
    def file_size_mb(self) -> float:
        if self.active_key == "ensemble":
            return sum(c.size_mb for c in self.ensemble_classifiers)
        if self.classifier:
            return self.classifier.size_mb
        return 0.0

    def model_embed(self) -> discord.Embed:
        embed = discord.Embed(title="🧠 AI Model Configuration & Live Switch", color=COLOR_INFO)
        embed.add_field(name="Active Selection", value=f"**{self.display_name}**", inline=False)
        embed.add_field(name="Resolution", value=f"`{self.resolution_str}`", inline=True)
        embed.add_field(name="Model Size", value=f"`{self.file_size_mb:.1f} MB`", inline=True)
        ms_val = f"`{self.last_ms:.0f} ms`" if self.last_ms > 0 else "*Pending first run*"
        embed.add_field(name="Last Latency (Pi 4B)", value=ms_val, inline=True)

        all_models = self.discover_models()
        model_list_str = []
        for m in all_models:
            active_marker = " ◀ **ACTIVE**" if m.name == self.active_key else ""
            size = m.stat().st_size / (1024 * 1024) if m.exists() else 0.0
            model_list_str.append(f"• `{m.name}` ({size:.1f} MB){active_marker}")
        if len(all_models) >= 2:
            ens_active = " ◀ **ACTIVE**" if self.active_key == "ensemble" else ""
            model_list_str.append(f"• `🧠 Ensemble Mode (Dual-Brain)`{ens_active}")

        embed.add_field(name="Available Models in AI/", value="\n".join(model_list_str) or "None found", inline=False)
        embed.set_footer(text="Select a model below to hot-swap live on the Raspberry Pi without restarting.")
        return embed


# --------------------------------------------------------------------------------------
# Image helpers (everything stays in RAM)
# --------------------------------------------------------------------------------------
def _draw_roi_brackets(bgr: np.ndarray, box: tuple[float, float, float, float]) -> None:
    """Draw sleek cyan corner brackets indicating the Nozzle ROI focus zone."""
    h, w = bgr.shape[:2]
    x1, y1 = int(round(box[0] * w)), int(round(box[1] * h))
    x2, y2 = int(round(box[2] * w)), int(round(box[3] * h))
    arm = max(14, min(x2 - x1, y2 - y1) // 8)
    col = (235, 210, 50)  # cyan-gold in BGR
    th = max(2, w // 480)
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(bgr, (px, py), (px + dx * arm, py), col, th, cv2.LINE_AA)
        cv2.line(bgr, (px, py), (px, py + dy * arm), col, th, cv2.LINE_AA)


def encode_jpeg(rgb: np.ndarray, quality: int, pred: Optional[Prediction] = None,
                threshold: float = 1.0, caption: str = "",
                roi_overlay: Optional[tuple[float, float, float, float]] = None) -> bytes:
    """RGB frame -> JPEG bytes, with an optional label banner and Nozzle ROI corner brackets."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if roi_overlay is not None:
        _draw_roi_brackets(bgr, roi_overlay)
    if pred is not None or caption:
        h, w = bgr.shape[:2]
        bar_h = max(28, h // 16)
        roi = bgr[:bar_h]
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, dst=roi)  # translucent banner
        scale = bar_h / 42
        if pred is not None:
            alarm = pred.is_alarm(threshold)
            suspicious = not alarm and pred.failure_conf >= 0.50
            if alarm:
                color = (60, 60, 235)  # Red (BGR)
                text = f"{ERROR_CLASS} {pred.failure_conf * 100:.1f}%"
            elif suspicious:
                color = (0, 190, 255)  # Amber (BGR)
                text = f"{ERROR_CLASS} {pred.failure_conf * 100:.1f}%"
            else:
                color = (90, 210, 90)  # Green (BGR)
                text = f"{OK_CLASS} {pred.ok_conf * 100:.1f}%"
            cv2.putText(bgr, text, (8, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        max(1, int(scale * 2)), cv2.LINE_AA)
        stamp = caption or datetime.now().strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, 1)
        cv2.putText(bgr, stamp, (w - tw - 8, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8,
                    (230, 230, 230), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def jpeg_file(data: bytes, name: str) -> discord.File:
    return discord.File(io.BytesIO(data), filename=name)


def progress_bar(frac: float, width: int = 14) -> str:
    frac = min(1.0, max(0.0, frac))
    n = round(frac * width)
    return "█" * n + "░" * (width - n)


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def ts(t: Optional[float] = None, style: str = "R") -> str:
    return f"<t:{int(t if t is not None else time.time())}:{style}>"


def read_pi_stats() -> dict[str, Any]:
    """Raspberry Pi health without psutil: /sys + /proc only."""
    stats: dict[str, Any] = {}
    try:
        stats["cpu_temp"] = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0])
        stats["ram_pct"] = 100 * (1 - mem["MemAvailable"] / mem["MemTotal"])
        stats["ram_total_mb"] = mem["MemTotal"] / 1024
        stats["ram_used_mb"] = (mem["MemTotal"] - mem["MemAvailable"]) / 1024
    except (OSError, KeyError, ValueError):
        pass
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                stats["bot_rss_mb"] = int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        stats["load"] = os.getloadavg()
    except OSError:
        pass
    return stats


# --------------------------------------------------------------------------------------
# Moonraker client
# --------------------------------------------------------------------------------------
class MoonrakerError(Exception):
    pass


class Moonraker:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, api_key: str = ""):
        self._s = session
        self.base = base_url
        self._headers = {"X-Api-Key": api_key} if api_key else {}
        self._objects: Optional[list[str]] = None
        self._meta_cache: dict[str, dict] = {}

    async def _call(self, method: str, path: str, timeout: float = 8.0) -> Any:
        try:
            async with self._s.request(method, self.base + path, headers=self._headers,
                                       timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                data = await r.json(content_type=None)
                if r.status >= 400:
                    msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
                    raise MoonrakerError(f"HTTP {r.status}: {msg or r.reason}")
                return data.get("result") if isinstance(data, dict) else data
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            raise MoonrakerError(f"{type(e).__name__}: {e}") from e

    async def query(self, *objects: str) -> dict[str, dict]:
        # Bare object names (no "=") -> Moonraker returns every attribute of the object.
        qs = "&".join(quote(o, safe="") for o in objects)
        res = await self._call("GET", f"/printer/objects/query?{qs}")
        return res.get("status", {})

    async def list_objects(self) -> list[str]:
        if self._objects is None:
            self._objects = (await self._call("GET", "/printer/objects/list")).get("objects", [])
        return self._objects

    async def server_info(self) -> dict:
        return await self._call("GET", "/server/info")

    async def printer_info(self) -> dict:
        return await self._call("GET", "/printer/info")

    async def print_state(self) -> str:
        return (await self.query("print_stats"))["print_stats"].get("state", "unknown")

    async def file_metadata(self, filename: str) -> dict:
        if not filename:
            return {}
        if filename not in self._meta_cache:
            try:
                self._meta_cache[filename] = await self._call(
                    "GET", f"/server/files/metadata?filename={quote(filename)}")
            except MoonrakerError:
                return {}
            if len(self._meta_cache) > 8:
                self._meta_cache.pop(next(iter(self._meta_cache)))
        return self._meta_cache[filename]

    async def _action(self, path: str) -> tuple[bool, str]:
        try:
            await self._call("POST", path, timeout=60.0)  # PAUSE/CANCEL macros can take a while
            return True, ""
        except MoonrakerError as e:
            return False, str(e)

    async def pause(self) -> tuple[bool, str]:
        return await self._action("/printer/print/pause")

    async def resume(self) -> tuple[bool, str]:
        return await self._action("/printer/print/resume")

    async def cancel(self) -> tuple[bool, str]:
        return await self._action("/printer/print/cancel")

    def invalidate_objects(self) -> None:
        self._objects = None


# --------------------------------------------------------------------------------------
# Camera: Crowsnest snapshot first, Picamera2 fallback
# --------------------------------------------------------------------------------------
class CameraError(Exception):
    pass


class Camera:
    def __init__(self, cfg: Config, session: aiohttp.ClientSession):
        self.cfg = cfg
        self._s = session
        self._picam = None
        self._picam_failed_at = 0.0
        self._lock = asyncio.Lock()
        self.last_source = ""

    async def grab(self) -> np.ndarray:
        """Return a fresh RGB uint8 frame."""
        async with self._lock:
            http_err = ""
            data = None
            try:
                async with self._s.get(self.cfg.snapshot_url,
                                       timeout=aiohttp.ClientTimeout(total=self.cfg.snapshot_timeout)) as r:
                    if r.status == 200:
                        data = await r.read()
                    else:
                        http_err = f"HTTP {r.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                http_err = type(e).__name__
            if data:
                frame = await asyncio.to_thread(self._decode, data)
                if frame is not None:
                    self.last_source = "crowsnest"
                    if self._picam is not None:  # Crowsnest is back -> free the sensor
                        await asyncio.to_thread(self._close_picam)
                    return frame
                http_err = "undecodable JPEG"

            frame = await asyncio.to_thread(self._grab_picam)
            if frame is None:
                raise CameraError(f"snapshot failed ({http_err or 'no data'}) and Picamera2 unavailable")
            self.last_source = "picamera2"
            return frame

    def _decode(self, data: bytes) -> Optional[np.ndarray]:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        if w > self.cfg.snapshot_max_width:  # keep RAM small for 1080p+ streams
            nh = int(h * self.cfg.snapshot_max_width / w)
            bgr = cv2.resize(bgr, (self.cfg.snapshot_max_width, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.flip(rgb, -1) if self.cfg.flip_snapshot_180 else rgb

    def _grab_picam(self) -> Optional[np.ndarray]:
        if self._picam is None:
            if time.monotonic() - self._picam_failed_at < 60:  # back-off after a failure
                return None
            try:
                from picamera2 import Picamera2  # apt: python3-picamera2
                cam = Picamera2()
                conf = cam.create_preview_configuration(
                    main={"size": (self.cfg.picam_width, self.cfg.picam_height), "format": "RGB888"},
                    buffer_count=2)  # 2 small buffers instead of a full-res still pipeline
                cam.configure(conf)
                cam.start()
                try:  # manual macro focus on the nozzle (Camera Module 3)
                    cam.set_controls({"AfMode": 0, "LensPosition": self.cfg.lens_position})
                except Exception as e:
                    log.warning("Camera has no manual focus control: %s", e)
                time.sleep(1.0)  # let AE/AWB and the lens settle
                self._picam = cam
                log.info("Picamera2 fallback started (%dx%d, LensPosition=%.1f)",
                         self.cfg.picam_width, self.cfg.picam_height, self.cfg.lens_position)
            except Exception as e:  # ImportError, "Device or resource busy", ...
                self._picam_failed_at = time.monotonic()
                log.warning("Picamera2 unavailable: %s", e)
                self._close_picam()
                return None
        try:
            arr = self._picam.capture_array("main")
        except Exception as e:
            log.warning("Picamera2 capture failed: %s", e)
            self._close_picam()
            self._picam_failed_at = time.monotonic()
            return None
        rgb = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2RGB)  # Picamera2 "RGB888" is B,G,R in memory
        return cv2.flip(rgb, -1) if self.cfg.flip_camera_180 else rgb

    def _close_picam(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
            log.info("Picamera2 released")

    async def release(self) -> None:
        """Free the sensor + its DMA buffers (called when the printer goes idle)."""
        if self._picam is not None:
            async with self._lock:
                await asyncio.to_thread(self._close_picam)


# --------------------------------------------------------------------------------------
# Embeds
# --------------------------------------------------------------------------------------
def prob_lines(pred: Prediction, threshold: float) -> str:
    active = ERROR_CLASS if pred.is_alarm(threshold) else pred.label
    lines = []
    for name in CLASSES:
        p = pred.probs.get(name, 0.0)
        mark = " ◀" if name == active else ""
        lines.append(f"{CLASS_EMOJI.get(name, '•')} `{progress_bar(p)}` **{p * 100:5.1f}%** {name}{mark}")
    view_note = f" · focus: {pred.view}" if pred.view else ""
    lines.append(f"-# Alert threshold: {threshold * 100:.0f}%{view_note}")
    return "\n".join(lines)


def set_status_field(embed: discord.Embed, value: str, name: str = "Status") -> None:
    for i, f in enumerate(embed.fields):
        if f.name == name:
            embed.set_field_at(i, name=name, value=value, inline=False)
            return
    embed.add_field(name=name, value=value, inline=False)


@dataclass
class PrintSummary:
    state: str = "unknown"
    filename: str = ""
    progress: float = 0.0
    elapsed: Optional[float] = None
    remaining: Optional[float] = None
    slicer_remaining: Optional[float] = None
    layer: str = "—"
    filament_m: float = 0.0
    hotend: tuple[float, float] = (0.0, 0.0)
    bed: tuple[float, float] = (0.0, 0.0)
    message: str = ""
    sensors: list[str] = field(default_factory=list)

    @property
    def high_temp(self) -> bool:
        return max(self.hotend) > HIGH_TEMP_C


def summarize(status: dict[str, dict], meta: dict) -> PrintSummary:
    ps = status.get("print_stats", {})
    vsd = status.get("virtual_sdcard", {})
    ds = status.get("display_status", {})
    ext = status.get("extruder", {})
    bed = status.get("heater_bed", {})
    s = PrintSummary(state=ps.get("state", "unknown"), filename=ps.get("filename", "") or "",
                     message=ps.get("message", "") or "")
    s.progress = float(ds.get("progress") or vsd.get("progress") or 0.0)  # M73 if slicer emits it
    dur = float(ps.get("print_duration") or 0.0)
    s.elapsed = float(ps.get("total_duration") or 0.0) or None
    if s.progress > 0.01 and dur > 0:
        s.remaining = dur / s.progress - dur
    est = meta.get("estimated_time")
    if est:
        s.slicer_remaining = max(0.0, float(est) - dur)
    info = ps.get("info") or {}
    cur, total = info.get("current_layer"), info.get("total_layer") or meta.get("layer_count")
    if cur is not None:
        s.layer = f"{cur} / {total}" if total else str(cur)
    s.filament_m = float(ps.get("filament_used") or 0.0) / 1000
    s.hotend = (float(ext.get("temperature") or 0.0), float(ext.get("target") or 0.0))
    s.bed = (float(bed.get("temperature") or 0.0), float(bed.get("target") or 0.0))
    for name, obj in status.items():
        if name.startswith("temperature_sensor "):
            s.sensors.append(f"🌡️ {name.split(' ', 1)[1]}: **{obj.get('temperature', 0):.1f} °C**")
        elif name.startswith("temperature_fan "):
            spd = float(obj.get("speed") or 0.0) * 100
            s.sensors.append(f"🌀 {name.split(' ', 1)[1]}: **{obj.get('temperature', 0):.1f} °C** "
                             f"→ {obj.get('target', 0):.0f} °C · fan {spd:.0f}%")
    return s


def temp_str(actual: float, target: float) -> str:
    return f"**{actual:.1f}** / {target:.0f} °C" if target else f"**{actual:.1f}** °C (off)"


# --------------------------------------------------------------------------------------
# Views (all buttons are persistent: they keep working after a bot restart)
# --------------------------------------------------------------------------------------
class GuardedView(discord.ui.View):
    """Base view: only the configured owner(s) may press buttons; timeout=None (persistent)."""

    def __init__(self, bot: "TDRBot"):
        super().__init__(timeout=None)
        self.bot = bot

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await self.bot.check_auth(interaction)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item) -> None:
        log.exception("Button %s failed", getattr(item, "custom_id", item), exc_info=error)
        await self.bot.safe_reply(interaction, f"⚠️ Error: `{error}`")

    def _button(self, custom_id: str, label: str, emoji: str, style: discord.ButtonStyle, cb) -> None:
        b = discord.ui.Button(label=label, emoji=emoji, style=style, custom_id=custom_id)
        b.callback = cb
        self.add_item(b)


class AlertView(GuardedView):
    STOP, PAUSE, RESUME, IGNORE = "tdr:alert:stop", "tdr:alert:pause", "tdr:alert:resume", "tdr:alert:ignore"
    PHASES = {
        "detected": (STOP, PAUSE, IGNORE),
        "paused": (RESUME, STOP),
        "running": (PAUSE, STOP),
        "all": (STOP, PAUSE, RESUME, IGNORE),  # registration only
    }

    def __init__(self, bot: "TDRBot", phase: str = "detected"):
        super().__init__(bot)
        spec = {
            self.STOP: ("Stop Print", "🛑", discord.ButtonStyle.danger, "stop"),
            self.PAUSE: ("Pause Print", "⏸️", discord.ButtonStyle.primary, "pause"),
            self.RESUME: ("Resume Print", "▶️", discord.ButtonStyle.success, "resume"),
            self.IGNORE: ("Ignore (False Alarm)", "✅", discord.ButtonStyle.secondary, "ignore"),
        }
        for cid in self.PHASES[phase]:
            label, emoji, style, action = spec[cid]
            self._button(cid, label, emoji, style, self._make_cb(action))

    def _make_cb(self, action: str):
        async def cb(interaction: discord.Interaction):
            await self.bot.handle_alert_action(interaction, action)
        return cb


class ModelSelect(discord.ui.Select):
    def __init__(self, bot: "TDRBot"):
        self.bot = bot
        models = bot.model_manager.discover_models()
        active = bot.model_manager.active_key

        options = []
        for m in models:
            size_mb = m.stat().st_size / (1024 * 1024) if m.exists() else 0.0
            is_eff = "efficientnet" in m.name.lower()
            res = "300x300" if is_eff else "224x224"
            desc = f"{res} · {size_mb:.1f} MB"
            emoji = "🧠" if is_eff else ("⚡" if "mobilenet" in m.name.lower() or "cambridge" in m.name.lower() else "📦")
            options.append(discord.SelectOption(
                label=m.name[:100],
                value=m.name,
                description=desc[:100],
                emoji=emoji,
                default=(m.name == active)
            ))

        if len(models) >= 2:
            options.append(discord.SelectOption(
                label="🧠 Ensemble Mode (Dual-Brain · Recommended)",
                value="ensemble",
                description="Motion-robust MobileNetV2 + EfficientNetV2S consensus",
                emoji="🔮",
                default=(active == "ensemble")
            ))

        if not options:
            options.append(discord.SelectOption(
                label="No models available",
                value="none",
                description="Add .tflite to AI/ folder"
            ))

        super().__init__(
            placeholder="Choose active AI model...",
            min_values=1,
            max_values=1,
            options=options,
            custom_id="tdr:model:select"
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        choice = self.values[0]
        if choice == "none":
            await interaction.followup.send("⚠️ No models available to activate.", ephemeral=True)
            return
        ok, msg = await self.bot.model_manager.switch_model(choice)
        if ok:
            self.bot.clear_streak()
        embed = self.bot.model_manager.model_embed()
        if not ok:
            embed.description = f"⚠️ {msg}"
        view = ModelSelectView(self.bot)
        await interaction.edit_original_response(embed=embed, view=view)


class ModelSelectView(GuardedView):
    def __init__(self, bot: "TDRBot"):
        super().__init__(bot)
        self.add_item(ModelSelect(bot))


class DashboardView(GuardedView):
    CHECK, AI, MODEL, PAUSE, CANCEL, REFRESH = (
        "tdr:dash:check", "tdr:dash:ai", "tdr:dash:model",
        "tdr:dash:pause", "tdr:dash:cancel", "tdr:dash:refresh"
    )

    def __init__(self, bot: "TDRBot", printer_state: str = ""):
        super().__init__(bot)
        ai_on = bot.settings.enabled
        self._button(self.CHECK, "Check Print", "📸", discord.ButtonStyle.primary, bot.handle_check)
        self._button(self.AI, f"AI: {'ON' if ai_on else 'OFF'}", "🤖",
                     discord.ButtonStyle.success if ai_on else discord.ButtonStyle.secondary, self._toggle_ai)
        self._button(self.MODEL, "Switch Model", "🧠", discord.ButtonStyle.secondary, self._switch_model)
        if printer_state == "paused":
            self._button(self.PAUSE, "Resume", "▶️", discord.ButtonStyle.success, self._pause_resume)
        else:
            self._button(self.PAUSE, "Pause", "⏸️", discord.ButtonStyle.secondary, self._pause_resume)
        self._button(self.CANCEL, "Emergency Cancel", "🛑", discord.ButtonStyle.danger, self._cancel)
        self._button(self.REFRESH, "Refresh", "🔄", discord.ButtonStyle.secondary, self._refresh)

    async def _switch_model(self, interaction: discord.Interaction):
        embed = self.bot.model_manager.model_embed()
        view = ModelSelectView(self.bot)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    async def _refresh(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self.bot.edit_dashboard(interaction)

    async def _toggle_ai(self, interaction: discord.Interaction):
        await interaction.response.defer()
        self.bot.set_ai_enabled(not self.bot.settings.enabled)
        await self.bot.edit_dashboard(interaction)

    async def _pause_resume(self, interaction: discord.Interaction):
        await interaction.response.defer()
        state = await self.bot.safe_print_state()
        if state == "printing":
            ok, err = await self.bot.moonraker.pause()
            note = "⏸️ Print paused." if ok else f"❌ Pause failed: `{err}`"
        elif state == "paused":
            ok, err = await self.bot.moonraker.resume()
            note = "▶️ Print resumed." if ok else f"❌ Resume failed: `{err}`"
            if ok:
                self.bot.on_resumed()
        else:
            note = f"ℹ️ Nothing to pause/resume (printer is `{state}`)."
        await interaction.followup.send(note, ephemeral=True)
        await self.bot.edit_dashboard(interaction)

    async def _cancel(self, interaction: discord.Interaction):
        await interaction.response.send_message(
            "⚠️ **Cancel the current print?** This cannot be undone.",
            view=ConfirmCancelView(self.bot), ephemeral=True)


class ConfirmCancelView(discord.ui.View):
    """Short-lived (non-persistent) confirmation for the dashboard's cancel button."""

    def __init__(self, bot: "TDRBot"):
        super().__init__(timeout=30)
        self.bot = bot

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await self.bot.check_auth(interaction)

    @discord.ui.button(label="Yes, cancel print", emoji="🛑", style=discord.ButtonStyle.danger)
    async def yes(self, interaction: discord.Interaction, _: discord.ui.Button):
        await interaction.response.edit_message(content="⏳ Cancelling…", view=None)
        state = await self.bot.safe_print_state()
        if state not in ("printing", "paused"):
            await interaction.edit_original_response(content=f"ℹ️ No active print (printer is `{state}`).")
            return
        ok, err = await self.bot.moonraker.cancel()
        await interaction.edit_original_response(content="🛑 Print cancelled." if ok else f"❌ Cancel failed: `{err}`")
        self.stop()

    @discord.ui.button(label="No", style=discord.ButtonStyle.secondary)
    async def no(self, interaction: discord.Interaction, _: discord.ui.Button):
        await interaction.response.edit_message(content="👍 Kept printing.", view=None)
        self.stop()


class AIPanelView(GuardedView):
    TOGGLE, MODE, CHECK, MODEL = "tdr:ai:toggle", "tdr:ai:mode", "tdr:ai:check", "tdr:ai:model"

    def __init__(self, bot: "TDRBot"):
        super().__init__(bot)
        s = bot.settings
        if s.enabled:
            self._button(self.TOGGLE, "Turn AI OFF", "🔴", discord.ButtonStyle.danger, self._toggle)
        else:
            self._button(self.TOGGLE, "Turn AI ON", "🟢", discord.ButtonStyle.success, self._toggle)
        if s.mode == "supervised":
            self._button(self.MODE, "Switch to Autonomous", "🤖", discord.ButtonStyle.secondary, self._mode)
        else:
            self._button(self.MODE, "Switch to Supervised", "🧑", discord.ButtonStyle.secondary, self._mode)
        self._button(self.CHECK, "Check Print Now", "📸", discord.ButtonStyle.primary, bot.handle_check)
        self._button(self.MODEL, "Switch Model", "🧠", discord.ButtonStyle.secondary, self._switch_model)

    async def _switch_model(self, interaction: discord.Interaction):
        embed = self.bot.model_manager.model_embed()
        view = ModelSelectView(self.bot)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    async def _toggle(self, interaction: discord.Interaction):
        self.bot.set_ai_enabled(not self.bot.settings.enabled)
        await self.bot.respond_panel(interaction, edit=True)

    async def _mode(self, interaction: discord.Interaction):
        self.bot.set_mode("autonomous" if self.bot.settings.mode == "supervised" else "supervised")
        await self.bot.respond_panel(interaction, edit=True)


# --------------------------------------------------------------------------------------
# Bot
# --------------------------------------------------------------------------------------
@dataclass
class ActiveAlert:
    message: Optional[discord.Message]
    jpeg: bytes
    pred: Prediction
    created: float = field(default_factory=time.time)
    timeout_task: Optional[asyncio.Task] = None


class GuardedTree(app_commands.CommandTree):
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await self.client.check_auth(interaction)  # type: ignore[attr-defined]

    async def on_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        if isinstance(error, app_commands.CheckFailure):
            return
        log.exception("Slash command failed", exc_info=error)
        await self.client.safe_reply(interaction, f"⚠️ Error: `{getattr(error, 'original', error)}`")  # type: ignore


class TDRBot(discord.Client):
    def __init__(self, cfg: Config):
        intents = discord.Intents.default()
        # Trim gateway traffic we never use (less CPU + RAM on the Pi).
        intents.typing = False
        intents.presences = False
        intents.members = False
        intents.voice_states = False
        intents.invites = False
        intents.emojis_and_stickers = False
        intents.integrations = False
        intents.webhooks = False
        super().__init__(
            intents=intents,
            member_cache_flags=discord.MemberCacheFlags.none(),
            max_messages=None,
            chunk_guilds_at_startup=False,
        )
        self.cfg = cfg
        # Slash commands usable both in your server and directly in the bot's DM.
        self.tree = GuardedTree(self, allowed_contexts=app_commands.AppCommandContext(guild=True, dm_channel=True))
        self.settings = AISettings.load(cfg)
        self.session: Optional[aiohttp.ClientSession] = None
        self.moonraker: Moonraker = None  # type: ignore[assignment]
        self.camera: Camera = None  # type: ignore[assignment]
        self.model_manager = ModelManager(self)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tflite")
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._owner: Optional[discord.User] = None
        # runtime AI state
        self.streak: list[Prediction] = []
        self.window: list[Prediction] = []
        self.snooze_until = 0.0
        self.alert: Optional[ActiveAlert] = None
        self.last_pred: Optional[Prediction] = None
        self.printer_state = "unknown"
        self.printer_z: Optional[float] = None
        self.prep_reason = ""
        self._last_xyz: Optional[tuple[float, float, float]] = None
        self._last_xyz_ts = 0.0
        self._still_polls = 0
        self.inference_count = 0

    @property
    def classifier(self) -> Optional[TFLiteClassifier]:
        return self.model_manager.classifier

    @property
    def model_error(self) -> str:
        return self.model_manager.model_error

    # ------------------------------------------------------------------ lifecycle
    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=4, ttl_dns_cache=300))
        self.moonraker = Moonraker(self.session, self.cfg.moonraker_url, self.cfg.moonraker_api_key)
        self.camera = Camera(self.cfg, self.session)

        await self.model_manager.initialize()

        register_commands(self)
        for view in (AlertView(self, "all"), DashboardView(self), AIPanelView(self), ModelSelectView(self)):
            self.add_view(view)  # persistent dispatch by custom_id
        synced = await self.tree.sync()
        log.info("Synced %d slash commands", len(synced))

        self._tasks = [asyncio.create_task(self.ai_monitor_loop(), name="ai-monitor"),
                       asyncio.create_task(self.printer_watch_loop(), name="printer-watch")]

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id %s)", self.user, self.user.id if self.user else "?")
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching,
                                                             name="the Ender 3 nozzle"))
        if self.cfg.notify_startup and not getattr(self, "_announced", False):
            self._announced = True
            embed = self.panel_embed()
            embed.title = "🟢 TDR AI bot online"
            await self.notify(embed=embed, view=AIPanelView(self))

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.camera:
            await self.camera.release()
        await self.model_manager.close()
        await super().close()
        if self.session:
            await self.session.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ helpers
    def is_authorized(self, user_id: int) -> bool:
        return user_id == self.cfg.user_id or user_id in self.cfg.extra_user_ids

    async def check_auth(self, interaction: discord.Interaction) -> bool:
        if self.is_authorized(interaction.user.id):
            return True
        await self.safe_reply(interaction, "⛔ You are not allowed to control this printer.")
        return False

    async def safe_reply(self, interaction: discord.Interaction, content: str) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content, ephemeral=True)
            else:
                await interaction.response.send_message(content, ephemeral=True)
        except discord.HTTPException:
            pass

    async def safe_print_state(self) -> str:
        """Query Moonraker print state, live Z height, temperatures, extrusion & toolhead motion.

        Populates ``self.prep_reason`` whenever Klipper reports ``state == "printing"`` while the
        machine is still warming up (bed/hotend heating), homing / probing BLTouch / purging
        (``filament_used < 35 mm``), or standing still so background monitoring never fires false
        alarms before the toolhead is hot and actively printing.
        """
        try:
            res = await self.moonraker.query("print_stats", "extruder", "heater_bed", "gcode_move", "motion_report")
            ps = res.get("print_stats", {})
            st = ps.get("state", "unknown")
            mr = res.get("motion_report", {})
            gm = res.get("gcode_move", {})
            has_mr = bool(mr and isinstance(mr.get("live_position"), (list, tuple)))
            pos = mr.get("live_position") if has_mr else gm.get("gcode_position")
            xyz: Optional[tuple[float, float, float]] = None
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
                self.printer_z = xyz[2]

            if st != "printing":
                self.prep_reason = ""
                self._last_xyz = None
                self._last_xyz_ts = 0.0
                self._still_polls = 0
                return st

            ext = res.get("extruder", {})
            bed = res.get("heater_bed", {})
            hot_temp = float(ext.get("temperature") or 0.0)
            hot_target = float(ext.get("target") or 0.0)
            bed_temp = float(bed.get("temperature") or 0.0)
            bed_target = float(bed.get("target") or 0.0)
            filament_mm = float(ps.get("filament_used") or 0.0)
            live_vel = float(mr.get("live_velocity") or 0.0) if has_mr else 0.0

            now = time.monotonic()
            first_sample = self._last_xyz is None
            if xyz is not None:
                if self._last_xyz is None:
                    self._last_xyz = xyz
                    self._last_xyz_ts = now
                elif now - self._last_xyz_ts >= 1.8:
                    dt = now - self._last_xyz_ts
                    lx, ly, lz = self._last_xyz
                    obs_vel = math.sqrt((xyz[0] - lx) ** 2 + (xyz[1] - ly) ** 2 + (xyz[2] - lz) ** 2) / dt
                    self._last_xyz = xyz
                    self._last_xyz_ts = now
                    if obs_vel < 0.25 and (not has_mr or live_vel < 0.35):
                        self._still_polls += 1
                    else:
                        self._still_polls = 0

            min_still = 2 if has_mr else 5
            if bed_target >= 35.0 and bed_temp < bed_target - 3.5:
                self.prep_reason = f"heating bed ({bed_temp:.0f}/{bed_target:.0f} °C)"
            elif hot_target < 160.0:
                self.prep_reason = f"pre-print setup (hotend target {hot_target:.0f} °C)"
            elif hot_temp < hot_target - 5.0:
                self.prep_reason = f"heating hotend ({hot_temp:.0f}/{hot_target:.0f} °C)"
            elif filament_mm < 35.0:
                self.prep_reason = f"homing / prime line ({filament_mm:.0f} mm extruded)"
            elif self._still_polls >= min_still or (first_sample and has_mr and live_vel < 0.25):
                self.prep_reason = "toolhead stationary (waiting for movement)"
            else:
                self.prep_reason = ""
            return st
        except MoonrakerError:
            return "offline"

    @staticmethod
    def _untrack(view: Optional[discord.ui.View]) -> None:
        # The globally registered persistent view already handles these custom_ids;
        # stopping the per-message copy avoids an ever-growing view store on long uptimes.
        if view is not None:
            view.stop()

    def _wake_loop(self) -> None:
        self._wake.set()

    async def _sleep(self, seconds: float) -> None:
        """Sleep that is interrupted instantly by settings changes (AI on/off, mode, ...)."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    async def infer(self, frame: np.ndarray) -> Prediction:
        if not self.model_manager.is_ready():
            raise RuntimeError(self.model_error or "model not loaded")
        pred = await asyncio.get_running_loop().run_in_executor(self._executor, self.model_manager.predict, frame)
        pred.source = self.camera.last_source
        self.last_pred = pred
        self.inference_count += 1
        return pred

    async def encode(self, frame: np.ndarray, pred: Optional[Prediction] = None, caption: str = "",
                     show_roi: bool = False) -> bytes:
        roi_overlay = (parse_roi_box(self.settings.roi_box)
                       if show_roi and self.settings.roi_mode in ("multi", "roi") else None)
        return await asyncio.to_thread(encode_jpeg, frame, self.cfg.jpeg_quality, pred,
                                       self.settings.threshold, caption, roi_overlay)

    def clear_streak(self) -> None:
        self.streak.clear()
        self.window.clear()

    # ------------------------------------------------------------------ settings
    def set_ai_enabled(self, on: bool) -> None:
        self.settings.enabled = on
        self.settings.save()
        self.clear_streak()
        log.info("AI monitoring %s", "ENABLED" if on else "DISABLED")
        self._wake_loop()

    def set_mode(self, mode: str) -> None:
        self.settings.mode = mode
        self.settings.save()
        log.info("AI mode -> %s", mode)
        self._wake_loop()

    def on_resumed(self) -> None:
        """After a resume: forget old evidence and give the extruder a few seconds to recover."""
        self.clear_streak()
        self.snooze_until = max(self.snooze_until, time.time() + min(30, self.cfg.snooze_seconds))

    # ------------------------------------------------------------------ notifications
    async def owner(self) -> discord.User:
        if self._owner is None:
            self._owner = self.get_user(self.cfg.user_id) or await self.fetch_user(self.cfg.user_id)
        return self._owner

    async def notify(self, *, content: Optional[str] = None, embed: Optional[discord.Embed] = None,
                     jpeg: Optional[bytes] = None, filename: str = "frame.jpg",
                     view: Optional[discord.ui.View] = None) -> Optional[discord.Message]:
        """DM the owner; fall back to DISCORD_ALERT_CHANNEL_ID if DMs are closed."""
        def kwargs() -> dict:
            kw: dict[str, Any] = {"content": content, "embed": embed}
            if jpeg is not None:
                kw["file"] = jpeg_file(jpeg, filename)  # a discord.File is single-use -> rebuild
            if view is not None:
                kw["view"] = view
            return kw

        msg = None
        try:
            user = await self.owner()
            msg = await user.send(**kwargs())
        except (discord.Forbidden, discord.HTTPException, discord.NotFound) as e:
            log.warning("DM to %s failed (%s) – trying alert channel", self.cfg.user_id, e)
            if self.cfg.alert_channel_id:
                try:
                    ch = self.get_channel(self.cfg.alert_channel_id) or await self.fetch_channel(self.cfg.alert_channel_id)
                    kw = kwargs()
                    kw["content"] = f"<@{self.cfg.user_id}> " + (content or "")
                    msg = await ch.send(**kw)  # type: ignore[union-attr]
                except discord.HTTPException as e2:
                    log.error("Alert channel send failed: %s", e2)
        self._untrack(view)
        return msg

    # ------------------------------------------------------------------ embeds
    def panel_embed(self) -> discord.Embed:
        s = self.settings
        embed = discord.Embed(title="🤖 AI Print Monitor",
                              color=COLOR_OK if s.enabled else COLOR_IDLE)
        embed.add_field(name="Monitoring", value="🟢 **ON**" if s.enabled else "🔴 **OFF**")
        mode = "🧑 Supervised" if s.mode == "supervised" else f"🤖 Autonomous ({self.cfg.autonomous_action})"
        embed.add_field(name="Mode", value=mode)
        p_disp = (f"⏳ `{self.printer_state}` ({self.prep_reason})"
                  if self.printer_state == "printing" and self.prep_reason
                  else f"{STATE_EMOJI.get(self.printer_state, '❔')} `{self.printer_state}`")
        embed.add_field(name="Printer", value=p_disp)
        win_len = 1 if s.frames <= 1 else s.frames + 2
        embed.add_field(name="Threshold", value=f"{s.threshold * 100:.0f}% · {s.frames}/{win_len} frames "
                                                f"(~{s.frames * self.cfg.check_interval:.0f}s)")
        embed.add_field(name="Nozzle Focus (ROI)", value=f"`{s.roi_mode}` (`{s.roi_box}`)")
        embed.add_field(name="Safety timeout",
                        value=f"{s.supervised_timeout}s → auto-pause" if s.supervised_timeout else "disabled")
        snooze = ts(self.snooze_until) if self.snooze_until > time.time() else "—"
        embed.add_field(name="Snoozed until", value=snooze)
        if self.last_pred:
            p = self.last_pred
            lbl = ERROR_CLASS if p.is_alarm(s.threshold) else p.label
            conf = p.failure_conf if lbl == ERROR_CLASS else p.ok_conf
            v_str = f" · {p.view}" if p.view else ""
            embed.add_field(name="Last inference",
                            value=f"{CLASS_EMOJI.get(lbl, '•')} {lbl} {conf * 100:.1f}% · "
                                  f"{p.ms:.0f} ms · {p.source}{v_str}", inline=False)
        if self.model_manager.is_ready():
            embed.set_footer(text=f"{self.model_manager.summary_string} · {self.cfg.num_threads} threads · "
                                  f"{self.inference_count} inferences")
        else:
            embed.add_field(name="⚠️ Model", value=self.model_error or "not loaded", inline=False)
        if self.cfg.force_ai_when_idle:
            embed.description = "🧪 `FORCE_AI_WHEN_IDLE=true` – AI runs even when not printing."
        return embed

    async def respond_panel(self, interaction: discord.Interaction, edit: bool = False) -> None:
        view = AIPanelView(self)
        if edit:
            await interaction.response.edit_message(embed=self.panel_embed(), view=view)
        else:
            await interaction.response.send_message(embed=self.panel_embed(), view=view)
        self._untrack(view)

    def check_embed(self, pred: Optional[Prediction]) -> discord.Embed:
        if pred is None:
            return discord.Embed(title="📸 Snapshot (no AI)", color=COLOR_WARN,
                                 description=f"⚠️ {self.model_error or 'Model not loaded'}")
        thr = self.settings.threshold
        alarming = pred.is_alarm(thr)
        suspicious = not alarming and pred.failure_conf >= 0.50
        if alarming:
            color = COLOR_BAD
            title = f"🚨 {ERROR_CLASS} — {pred.failure_conf * 100:.1f}%"
            verdict = "🚨 **Above alert threshold** – print failure detected!"
        elif suspicious:
            color = COLOR_WARN
            title = f"⚠️ {ERROR_CLASS} — {pred.failure_conf * 100:.1f}%"
            verdict = "🟡 Suspicious, but below the alert threshold."
        else:
            color = COLOR_OK
            title = f"✅ {OK_CLASS} — {pred.ok_conf * 100:.1f}%"
            verdict = "✅ Print looks fine."
        embed = discord.Embed(title=title, color=color, timestamp=datetime.now().astimezone())
        embed.description = f"{verdict}\n\n{prob_lines(pred, thr)}"
        embed.add_field(name="AI monitor", value="🟢 ON" if self.settings.enabled else "🔴 OFF (on-demand check)")
        p_disp = (f"⏳ `{self.printer_state}` ({self.prep_reason})"
                  if self.printer_state == "printing" and self.prep_reason
                  else f"{STATE_EMOJI.get(self.printer_state, '❔')} `{self.printer_state}`")
        embed.add_field(name="Printer", value=p_disp)
        embed.add_field(name="ROI mode", value=f"`{self.settings.roi_mode}`", inline=True)
        embed.set_footer(text=f"{self.model_manager.summary_string} · Inference {pred.ms:.0f} ms · source: {pred.source or '?'}")
        embed.set_image(url="attachment://check.jpg")
        return embed

    async def fetch_summary(self, extra: tuple[str, ...] = ()) -> PrintSummary:
        objs = ("print_stats", "virtual_sdcard", "display_status", "extruder", "heater_bed") + extra
        status = await self.moonraker.query(*objs)
        meta = await self.moonraker.file_metadata(status.get("print_stats", {}).get("filename", ""))
        return summarize(status, meta)

    async def dashboard_embed(self) -> tuple[discord.Embed, str]:
        try:
            names = await self.moonraker.list_objects()
            extra = tuple(n for n in names if n.startswith(("temperature_sensor ", "temperature_fan ")))
            s = await self.fetch_summary(extra)
        except MoonrakerError as e:
            self.moonraker.invalidate_objects()
            state, detail = "offline", str(e)
            try:
                info = await self.moonraker.server_info()
                state = info.get("klippy_state", "offline")
                detail = "Klipper is not ready."
                pinfo = await self.moonraker.printer_info()
                detail = pinfo.get("state_message", detail)
            except MoonrakerError:
                pass
            embed = discord.Embed(title=f"{STATE_EMOJI.get(state, '⚫')} Ender 3 — {state}", color=COLOR_BAD,
                                  description=f"```{detail[:1000]}```")
            self._pi_field(embed)
            return embed, state

        st = s.state
        embed = discord.Embed(title=f"{STATE_EMOJI.get(st, '❔')} TDR Ender 3 — {st.capitalize()}",
                              color=STATE_COLOR.get(st, COLOR_IDLE), timestamp=datetime.now().astimezone())
        if s.filename:
            embed.add_field(name="📄 File", value=f"`{s.filename[:200]}`", inline=False)
        if st in ("printing", "paused"):
            embed.add_field(name="📈 Progress", value=f"`{progress_bar(s.progress, 18)}` **{s.progress * 100:.1f}%**",
                            inline=False)
            eta = fmt_duration(s.remaining)
            if s.slicer_remaining is not None:
                eta += f" · slicer {fmt_duration(s.slicer_remaining)}"
            embed.add_field(name="⏳ Remaining", value=eta)
            embed.add_field(name="⏱️ Elapsed", value=fmt_duration(s.elapsed))
            embed.add_field(name="🧱 Layer", value=s.layer)
        hot = temp_str(*s.hotend)
        if s.high_temp:
            hot += "\n🔥 **High-temp mode (PETG/ABS)**"
        embed.add_field(name="🔥 Hotend", value=hot)
        embed.add_field(name="🛏️ Bed", value=temp_str(*s.bed))
        embed.add_field(name="🧵 Filament", value=f"{s.filament_m:.2f} m")
        if s.sensors:
            embed.add_field(name="🌡️ Sensors", value="\n".join(s.sensors)[:1024], inline=False)
        self._pi_field(embed)
        ai = self.settings
        ai_val = f"{'🟢 ON' if ai.enabled else '🔴 OFF'} · {ai.mode} · {ai.threshold * 100:.0f}% · ROI `{ai.roi_mode}`\n" \
                 f"Model: **{self.model_manager.display_name}** (`{self.model_manager.resolution_str}`)"
        if self.last_pred:
            p = self.last_pred
            lbl = ERROR_CLASS if p.is_alarm(ai.threshold) else p.label
            conf = p.failure_conf if lbl == ERROR_CLASS else p.ok_conf
            ai_val += f"\nLast: {CLASS_EMOJI.get(lbl, '•')} {lbl} " \
                      f"{conf * 100:.0f}% ({p.ms:.0f} ms)"
        embed.add_field(name="🤖 AI", value=ai_val)
        if s.message:
            embed.add_field(name="💬 Message", value=s.message[:1024], inline=False)
        return embed, st

    @staticmethod
    def _pi_field(embed: discord.Embed) -> None:
        pi = read_pi_stats()
        parts = []
        if "cpu_temp" in pi:
            t = pi["cpu_temp"]
            parts.append(f"{'🥵' if t >= 75 else '🌡️'} CPU **{t:.1f} °C**")
        if "ram_pct" in pi:
            parts.append(f"🧠 RAM **{pi['ram_pct']:.0f}%** ({pi['ram_used_mb']:.0f}/{pi['ram_total_mb']:.0f} MB)")
        if "load" in pi:
            parts.append(f"⚙️ Load {pi['load'][0]:.2f}")
        if "bot_rss_mb" in pi:
            parts.append(f"🤖 Bot {pi['bot_rss_mb']:.0f} MB")
        embed.add_field(name="🍓 Raspberry Pi 4B", value="\n".join(parts) or "n/a")

    async def edit_dashboard(self, interaction: discord.Interaction) -> None:
        embed, state = await self.dashboard_embed()
        view = DashboardView(self, state)
        await interaction.edit_original_response(embed=embed, view=view)
        self._untrack(view)

    # ------------------------------------------------------------------ /check + button
    async def handle_check(self, interaction: discord.Interaction, note: str = "") -> None:
        await interaction.response.defer(thinking=True)
        try:
            frame = await self.camera.grab()
        except CameraError as e:
            prefix = f"{note}\n" if note else ""
            await interaction.followup.send(f"{prefix}📷 Camera error: `{e}`")
            return
        pred = None
        if self.model_manager.is_ready():
            self.printer_state = await self.safe_print_state()
            pred = await self.infer(frame)
        jpeg = await self.encode(frame, pred, show_roi=True)
        await interaction.followup.send(
            content=note or None,
            embed=self.check_embed(pred),
            file=jpeg_file(jpeg, "check.jpg"),
        )

    # ------------------------------------------------------------------ alerts
    def resolve_alert(self) -> None:
        a = self.alert
        self.alert = None
        if a and a.timeout_task and a.timeout_task is not asyncio.current_task():
            a.timeout_task.cancel()

    async def trigger_alert(self, frame: np.ndarray, pred: Prediction, printer_state: str) -> None:
        label = pred.failure_label
        confs = [p.failure_conf for p in self.streak] or [pred.failure_conf]
        jpeg = await self.encode(frame, pred, show_roi=(pred.view == "nozzle-roi"))
        try:
            summary: Optional[PrintSummary] = await self.fetch_summary()
        except MoonrakerError:
            summary = None
        printing = printer_state == "printing"
        log.warning("🚨 FAILURE %s (%.1f%%, %d frames, view=%s) – mode=%s", label, pred.failure_conf * 100,
                    len(confs), pred.view or "?", self.settings.mode)

        win_max = 1 if self.settings.frames <= 1 else self.settings.frames + 2
        frame_desc = ("1 frame" if len(confs) == 1
                      else f"{len(confs)} recent frames (window {self.settings.frames}/{win_max})")
        embed = discord.Embed(title=f"{CLASS_EMOJI.get(label, '🚨')} Print failure detected: {label}",
                              color=COLOR_BAD, timestamp=datetime.now().astimezone())
        embed.description = (f"The AI detected **{label}** in **{frame_desc}** "
                             f"(avg {np.mean(confs) * 100:.1f}%, last {pred.failure_conf * 100:.1f}%).\n\n"
                             f"{prob_lines(pred, self.settings.threshold)}")
        if summary:
            if summary.filename:
                embed.add_field(name="📄 File", value=f"`{summary.filename[:200]}`", inline=False)
            embed.add_field(name="📈 Progress", value=f"`{progress_bar(summary.progress)}` {summary.progress * 100:.1f}%")
            embed.add_field(name="🧱 Layer", value=summary.layer)
            embed.add_field(name="🔥 Hotend / 🛏️ Bed",
                            value=f"{summary.hotend[0]:.0f}/{summary.hotend[1]:.0f} °C · "
                                  f"{summary.bed[0]:.0f}/{summary.bed[1]:.0f} °C")
        embed.set_image(url="attachment://alert.jpg")
        embed.set_footer(text=f"{self.model_manager.summary_string} · Mode: {self.settings.mode} · {pred.ms:.0f} ms · source: {pred.source}")

        if self.settings.mode == "autonomous":
            if not printing:
                set_status_field(embed, f"🧪 Printer is `{printer_state}` – no action taken (test mode).")
                await self.notify(embed=embed, jpeg=jpeg, filename="alert.jpg")
                self.snooze_until = time.time() + self.cfg.snooze_seconds
                return
            action = self.cfg.autonomous_action
            ok, err = await (self.moonraker.pause() if action == "pause" else self.moonraker.cancel())
            if ok:
                if action == "pause":
                    set_status_field(embed, f"🤖 **Autonomous:** print **paused** automatically {ts()}.\n"
                                            "Inspect the nozzle, then resume or stop.")
                    await self.notify(embed=embed, jpeg=jpeg, filename="alert.jpg", view=AlertView(self, "paused"))
                else:
                    set_status_field(embed, f"🤖 **Autonomous:** print **cancelled** automatically {ts()}.")
                    await self.notify(embed=embed, jpeg=jpeg, filename="alert.jpg")
                return
            # Moonraker refused -> degrade to a supervised alert so a human still gets buttons.
            set_status_field(embed, f"⚠️ Auto-{action} **failed**: `{err}`\nPlease act manually:")
            msg = await self.notify(embed=embed, jpeg=jpeg, filename="alert.jpg", view=AlertView(self, "detected"))
            self.alert = ActiveAlert(msg, jpeg, pred)
            return

        # Supervised (human-in-the-loop)
        timeout = self.settings.supervised_timeout
        if timeout and printing:
            set_status_field(embed, f"⏱️ Waiting for you… **auto-pause {ts(time.time() + timeout)}** if no response.")
        else:
            set_status_field(embed, "⏳ Waiting for your decision.")
        msg = await self.notify(embed=embed, jpeg=jpeg, filename="alert.jpg", view=AlertView(self, "detected"))
        self.alert = ActiveAlert(msg, jpeg, pred)
        if timeout and printing:
            self.alert.timeout_task = asyncio.create_task(self._supervised_timeout(self.alert, timeout))

    async def _supervised_timeout(self, alert: ActiveAlert, timeout: int) -> None:
        await asyncio.sleep(timeout)
        if self.alert is not alert:
            return
        self.resolve_alert()
        state = await self.safe_print_state()
        if state != "printing":
            note, view = f"ℹ️ No response, but printer is already `{state}`.", None
        else:
            ok, err = await self.moonraker.pause()
            note = (f"⏱️ **No response in {timeout}s → print auto-paused** {ts()} to save filament."
                    if ok else f"⏱️ No response in {timeout}s, **auto-pause failed**: `{err}`")
            view = AlertView(self, "paused") if ok else AlertView(self, "detected")
            log.warning("Supervised timeout -> auto-pause %s", "OK" if ok else f"FAILED ({err})")
        if alert.message:
            embed = alert.message.embeds[0].copy() if alert.message.embeds else discord.Embed()
            embed.color = COLOR_WARN
            set_status_field(embed, note)
            try:
                await alert.message.edit(embed=embed, view=view)
                self._untrack(view)
            except discord.HTTPException as e:
                log.warning("Could not edit alert after timeout: %s", e)

    async def handle_alert_action(self, interaction: discord.Interaction, action: str) -> None:
        await interaction.response.defer()  # Moonraker macros can take > 3 s
        msg = interaction.message
        embed = msg.embeds[0].copy() if msg and msg.embeds else discord.Embed(title="Print alert")
        is_current = bool(self.alert and self.alert.message and msg and self.alert.message.id == msg.id)
        who = interaction.user.display_name
        state = await self.safe_print_state()
        view: Optional[discord.ui.View] = None

        if action == "stop":
            if state not in ("printing", "paused"):
                note = f"ℹ️ No active print to stop (printer is `{state}`)."
            else:
                ok, err = await self.moonraker.cancel()
                note = f"🛑 **Print cancelled** by {who} {ts()}." if ok else f"❌ Cancel failed: `{err}`"
                embed.color = COLOR_BAD if ok else COLOR_WARN
                if not ok:
                    view = AlertView(self, "paused" if state == "paused" else "detected")
            if is_current:
                self.resolve_alert()

        elif action == "pause":
            if state != "printing":
                note = f"ℹ️ Can't pause, printer is `{state}`."
                view = AlertView(self, "paused") if state == "paused" else None
            else:
                ok, err = await self.moonraker.pause()
                if ok:
                    note = f"⏸️ **Paused** by {who} {ts()}. Inspect the nozzle, then resume or stop."
                    embed.color = COLOR_WARN
                    view = AlertView(self, "paused")
                else:
                    note = f"❌ Pause failed: `{err}`"
                    view = AlertView(self, "detected")
            if is_current:
                self.resolve_alert()

        elif action == "resume":
            if state != "paused":
                note = f"ℹ️ Can't resume, printer is `{state}`."
                view = AlertView(self, "running") if state == "printing" else None
            else:
                ok, err = await self.moonraker.resume()
                if ok:
                    self.on_resumed()
                    note = f"▶️ **Resumed** by {who} {ts()}. AI watching again."
                    embed.color = COLOR_INFO
                    view = AlertView(self, "running")
                else:
                    note = f"❌ Resume failed: `{err}`"
                    view = AlertView(self, "paused")

        else:  # ignore / false alarm
            self.clear_streak()
            self.snooze_until = time.time() + self.cfg.snooze_seconds
            saved = ""
            if self.cfg.save_false_positives:
                jpeg = self.alert.jpeg if is_current and self.alert else None
                if jpeg is None and msg and msg.attachments:
                    try:
                        jpeg = await msg.attachments[0].read()
                    except discord.HTTPException:
                        jpeg = None
                if jpeg:
                    label = self.alert.pred.failure_label if is_current and self.alert else "unknown"
                    saved = await asyncio.to_thread(self._save_false_positive, jpeg, label)
            note = (f"✅ **False alarm** (marked by {who}). AI snoozed until {ts(self.snooze_until)}."
                    + (f"\n💾 Saved as `{saved}` for retraining." if saved else ""))
            embed.color = COLOR_OK
            if is_current:
                self.resolve_alert()

        set_status_field(embed, note)
        await interaction.edit_original_response(embed=embed, view=view)
        self._untrack(view)

    def _save_false_positive(self, jpeg: bytes, label: str) -> str:
        """Keep at most MAX_FALSE_POSITIVES images (oldest deleted first)."""
        folder = BASE_DIR / "false_positives"
        folder.mkdir(exist_ok=True)
        name = f"fp_{datetime.now():%Y%m%d_%H%M%S}_{label}.jpg"
        (folder / name).write_bytes(jpeg)
        files = sorted(folder.glob("fp_*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in files[:-self.cfg.max_false_positives]:
            old.unlink(missing_ok=True)
        return name

    # ------------------------------------------------------------------ background loops
    async def ai_monitor_loop(self) -> None:
        await self.wait_until_ready()
        log.info("AI monitor loop started")
        idle_logged = False
        prep_logged = ""
        last_cam_warn = 0.0
        while not self.is_closed():
            try:
                if not self.settings.enabled or not self.model_manager.is_ready():
                    self.clear_streak()
                    await self.camera.release()
                    await self._sleep(3600)  # woken instantly by /ai_toggle
                    continue

                # Zero-waste idle: only look through the camera while actively printing & moving.
                state = await self.safe_print_state()
                if state != "printing" and not self.cfg.force_ai_when_idle:
                    if not idle_logged:
                        log.info("Printer %s – AI sleeping", state)
                        idle_logged = True
                    prep_logged = ""
                    self.clear_streak()
                    await self.camera.release()
                    await self._sleep(self.cfg.idle_poll_interval)
                    continue
                if state == "printing" and self.prep_reason and not self.cfg.force_ai_when_idle:
                    phase = self.prep_reason.split("(")[0].strip()
                    if prep_logged != phase:
                        log.info("Printer preparing (%s) – AI waiting for active movement", self.prep_reason)
                        prep_logged = phase
                    self.clear_streak()
                    await self._sleep(self.cfg.check_interval)
                    continue
                if idle_logged or prep_logged:
                    log.info("Printer %s (hot & moving) – AI watching", state)
                    idle_logged = False
                    prep_logged = ""

                if self.alert is not None or time.time() < self.snooze_until:
                    await self._sleep(self.cfg.check_interval)
                    continue

                t0 = time.monotonic()
                frame = await self.camera.grab()
                pred = await self.infer(frame)
                log.debug("%s %.1f%% (fail %s %.1f%%, view=%s) %.0f ms", pred.label, pred.confidence * 100,
                          pred.failure_label, pred.failure_conf * 100, pred.view or "?", pred.ms)

                win_max = 1 if self.settings.frames <= 1 else self.settings.frames + 2
                self.window.append(pred)
                if len(self.window) > win_max:
                    self.window = self.window[-win_max:]
                hits = [p for p in self.window if p.is_alarm(self.settings.threshold)]
                self.streak = hits

                if pred.is_alarm(self.settings.threshold):
                    log.info("Failure frame %d/%d (window %d/%d, %s): %s %.1f%%",
                             len(hits), self.settings.frames, len(self.window), win_max,
                             pred.view or "?", pred.failure_label, pred.failure_conf * 100)
                    if len(hits) >= self.settings.frames:
                        await self.trigger_alert(frame, pred, state)
                        self.clear_streak()
                del frame
                await self._sleep(max(0.2, self.cfg.check_interval - (time.monotonic() - t0)))
            except asyncio.CancelledError:
                raise
            except CameraError as e:
                if time.monotonic() - last_cam_warn > 60:
                    log.warning("Camera: %s", e)
                    last_cam_warn = time.monotonic()
                await self._sleep(self.cfg.idle_poll_interval)
            except Exception:
                log.exception("AI loop error")
                await self._sleep(5)

    async def _klippy_state(self) -> tuple[str, dict, str]:
        """Return (effective_state, print_stats, error_message)."""
        info = await self.moonraker.server_info()
        ks = info.get("klippy_state", "disconnected")
        if ks == "ready":
            status = await self.moonraker.query("print_stats")
            ps = status.get("print_stats", {})
            return ps.get("state", "unknown"), ps, ps.get("message", "") if ps.get("state") == "error" else ""
        if ks in ("error", "shutdown"):
            msg = ""
            try:
                msg = (await self.moonraker.printer_info()).get("state_message", "")
            except MoonrakerError:
                pass
            return "error", {}, msg or f"Klipper {ks}"
        return ks, {}, ""  # startup / disconnected

    async def printer_watch_loop(self) -> None:
        """Lightweight state machine: completion / cancel / Klipper error DMs + alert bookkeeping."""
        await self.wait_until_ready()
        prev: Optional[str] = None
        last_ps: dict = {}
        while not self.is_closed():
            try:
                try:
                    state, ps, err = await self._klippy_state()
                except MoonrakerError:
                    state, ps, err = "offline", {}, ""
                self.printer_state = state
                if ps:
                    last_ps = ps
                if state != prev:
                    log.info("Printer state: %s -> %s", prev, state)
                    await self._on_transition(prev, state, ps or last_ps, err)
                    prev = state
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Watch loop error")
            await asyncio.sleep(self.cfg.watch_interval)

    async def _on_transition(self, prev: Optional[str], state: str, ps: dict, err: str) -> None:
        if state == "printing":
            self._last_xyz = None
            self._last_xyz_ts = 0.0
            self._still_polls = 0
            if prev == "paused":
                self.on_resumed()
            else:  # new print -> fresh start
                self.clear_streak()
                self.snooze_until = 0.0
            self._wake_loop()

        # An unresolved alert whose print stopped/paused elsewhere (Mainsail, console, ...)
        if self.alert and state != "printing":
            alert = self.alert
            self.resolve_alert()
            if alert.message:
                embed = alert.message.embeds[0].copy() if alert.message.embeds else discord.Embed()
                set_status_field(embed, f"ℹ️ Printer became `{state}` outside Discord {ts()}.")
                view = AlertView(self, "paused") if state == "paused" else None
                try:
                    await alert.message.edit(embed=embed, view=view)
                    self._untrack(view)
                except discord.HTTPException:
                    pass

        was_active = prev in ("printing", "paused")
        filename = ps.get("filename") or "?"
        if state == "complete" and was_active and self.cfg.notify_complete:
            embed = discord.Embed(title="🏁 Print complete!", color=COLOR_OK, timestamp=datetime.now().astimezone())
            embed.add_field(name="📄 File", value=f"`{filename}`", inline=False)
            embed.add_field(name="⏱️ Duration", value=fmt_duration(ps.get("total_duration")))
            embed.add_field(name="🧵 Filament", value=f"{float(ps.get('filament_used') or 0) / 1000:.2f} m")
            jpeg = None
            try:
                frame = await self.camera.grab()
                jpeg = await self.encode(frame, caption="FINISHED " + datetime.now().strftime("%H:%M"))
                embed.set_image(url="attachment://finished.jpg")
            except CameraError as e:
                embed.set_footer(text=f"No final snapshot: {e}")
            await self.notify(embed=embed, jpeg=jpeg, filename="finished.jpg")
        elif state == "cancelled" and was_active and self.cfg.notify_cancelled:
            embed = discord.Embed(title="🚫 Print cancelled", color=STATE_COLOR["cancelled"],
                                  description=f"`{filename}` · after {fmt_duration(ps.get('total_duration'))}")
            await self.notify(embed=embed)
        elif state == "error" and prev != "error" and self.cfg.notify_errors:
            embed = discord.Embed(title="🔴 Klipper error", color=COLOR_BAD, timestamp=datetime.now().astimezone(),
                                  description=f"```{(err or 'Unknown error')[:1800]}```")
            if was_active:
                embed.add_field(name="📄 Interrupted print", value=f"`{filename}`", inline=False)
            embed.set_footer(text="Check Mainsail – a FIRMWARE_RESTART may be required.")
            await self.notify(embed=embed)


# --------------------------------------------------------------------------------------
# Slash commands
# --------------------------------------------------------------------------------------
def register_commands(bot: TDRBot) -> None:
    tree = bot.tree

    @tree.command(name="ai_toggle", description="Turn AI failure monitoring ON/OFF (no argument = toggle)")
    @app_commands.describe(state="on / off (leave empty to toggle)")
    async def ai_toggle(interaction: discord.Interaction, state: Optional[Literal["on", "off"]] = None):
        bot.set_ai_enabled(not bot.settings.enabled if state is None else state == "on")
        await bot.respond_panel(interaction)

    @tree.command(name="ai_panel", description="Show the AI monitor control panel")
    async def ai_panel(interaction: discord.Interaction):
        await bot.respond_panel(interaction)

    @tree.command(name="mode", description="Supervised (ask me first) or Autonomous (act immediately)")
    @app_commands.describe(mode="Operating mode",
                           timeout="Supervised only: seconds before auto-pause if I don't answer (0 = never)")
    async def mode(interaction: discord.Interaction, mode: Literal["supervised", "autonomous"],
                   timeout: Optional[app_commands.Range[int, 0, 3600]] = None):
        if timeout is not None:
            bot.settings.supervised_timeout = timeout
        bot.set_mode(mode)  # also persists the timeout
        await bot.respond_panel(interaction)

    @tree.command(name="threshold", description="Set the alert confidence threshold and sliding-window debounce frames")
    @app_commands.describe(value="Confidence, e.g. 0.60 or 60 (%)",
                           frames="Failure frames required in sliding window (1-20)")
    async def threshold(interaction: discord.Interaction, value: app_commands.Range[float, 0.05, 99.9],
                        frames: Optional[app_commands.Range[int, 1, 20]] = None):
        v = value / 100 if value > 1 else value
        bot.settings.threshold = round(min(0.999, max(0.05, v)), 3)
        if frames is not None:
            bot.settings.frames = frames
        bot.settings.save()
        bot.clear_streak()
        await bot.respond_panel(interaction)

    @tree.command(name="roi", description="Configure Nozzle Focus (ROI) mode and crop box, then preview on camera")
    @app_commands.describe(
        mode="multi = max(Nozzle ROI, Full Frame) [recommended], roi = Nozzle crop only, full = Full camera frame",
        box="Normalized crop x1,y1,x2,y2 in 0..1 (e.g. 0.30,0.32,0.86,0.92) or 'reset'",
    )
    async def roi(interaction: discord.Interaction,
                  mode: Optional[Literal["multi", "roi", "full"]] = None,
                  box: Optional[str] = None):
        if box is not None:
            raw = box.strip().lower()
            if raw in ("reset", "default"):
                bot.settings.roi_box = DEFAULT_ROI_STR
            else:
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) != 4:
                    await interaction.response.send_message(
                        f"⚠️ Invalid `box` format. Expected `x1,y1,x2,y2` in `0..1` (e.g. `{DEFAULT_ROI_STR}`) or `reset`.",
                        ephemeral=True,
                    )
                    return
                try:
                    vals = tuple(float(p) for p in parts)
                except ValueError:
                    await interaction.response.send_message(
                        f"⚠️ Coordinates must be numbers between `0.0` and `1.0` (e.g. `{DEFAULT_ROI_STR}`).",
                        ephemeral=True,
                    )
                    return
                if not (0.0 <= vals[0] < vals[2] <= 1.0 and 0.0 <= vals[1] < vals[3] <= 1.0
                        and (vals[2] - vals[0]) >= 0.15 and (vals[3] - vals[1]) >= 0.15):
                    await interaction.response.send_message(
                        "⚠️ Box must satisfy `0 <= x1 < x2 <= 1`, `0 <= y1 < y2 <= 1`, and width/height >= `0.15`.",
                        ephemeral=True,
                    )
                    return
                bot.settings.roi_box = f"{vals[0]:.2f},{vals[1]:.2f},{vals[2]:.2f},{vals[3]:.2f}"
        if mode is not None:
            bot.settings.roi_mode = mode
        bot.settings.save()
        bot.clear_streak()
        note = f"🎯 **Nozzle Focus (ROI):** mode=`{bot.settings.roi_mode}` · box=`{bot.settings.roi_box}`"
        await bot.handle_check(interaction, note=note)

    @tree.command(name="check", description="Take a snapshot now and run the AI on it")
    async def check(interaction: discord.Interaction):
        await bot.handle_check(interaction)

    @tree.command(name="model", description="View active AI model and hot-swap between .tflite models or Ensemble Mode")
    async def model(interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        embed = bot.model_manager.model_embed()
        view = ModelSelectView(bot)
        await interaction.followup.send(embed=embed, view=view)
        bot._untrack(view)

    @tree.command(name="printer", description="Printer dashboard: temps, progress, Pi health, quick actions")
    async def printer(interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        embed, state = await bot.dashboard_embed()
        view = DashboardView(bot, state)
        await interaction.followup.send(embed=embed, view=view)
        bot._untrack(view)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
async def main() -> None:
    try:
        os.nice(10)
        logging.getLogger("discord_controller").info("Set process priority with os.nice(10) to favor Klipper/Moonraker")
    except Exception as e:
        logging.getLogger("discord_controller").debug("os.nice(10) notice: %s", e)
    cfg = Config.from_env()
    discord.utils.setup_logging(level=getattr(logging, cfg.log_level, logging.INFO), root=True)
    logging.getLogger("discord").setLevel(logging.WARNING)  # keep journald quiet
    bot = TDRBot(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):  # systemd stop -> graceful shutdown
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.close()))
    async with bot:
        await bot.start(cfg.token)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

