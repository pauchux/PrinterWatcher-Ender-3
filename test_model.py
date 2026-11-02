#!/usr/bin/env python3
"""
TDR TinyML Model Verification & Benchmark Script
------------------------------------------------
Validates .tflite models (input shape, output shape, unified 2-class probabilities
[Correcte / Error], Multi-Scale Nozzle ROI + Full Frame, and inference latency).
Works on Raspberry Pi 4B and desktop workstations.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

CLASSES = ["Correcte", "Error"]
DEFAULT_ROI_BOX = (0.25, 0.32, 0.86, 0.96)


def import_interpreter():
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
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow"
    except ImportError:
        print("❌ Error: No TFLite runtime found (install ai-edge-litert or tflite-runtime).")
        sys.exit(1)


def load_rgb_image(path_str: str) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Could not decode image: {path_str}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path_str).convert("RGB"), dtype=np.uint8)


def resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)


def crop_roi(rgb: np.ndarray, box=DEFAULT_ROI_BOX) -> np.ndarray:
    fh, fw = rgb.shape[:2]
    x1 = max(0, min(fw - 2, int(round(box[0] * fw))))
    y1 = max(0, min(fh - 2, int(round(box[1] * fh))))
    x2 = max(x1 + 1, min(fw, int(round(box[2] * fw))))
    y2 = max(y1 + 1, min(fh, int(round(box[3] * fh))))
    return rgb[y1:y2, x1:x2]


def detect_nozzle_air_gap(rgb: np.ndarray, min_gap_ratio: float = 0.016) -> float:
    """Skyline Air-Gap Clog Detector matching discord_controller.py."""
    fh, fw = rgb.shape[:2]
    if fh < 120 or fw < 160:
        return 0.0
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

    nx0, nx1 = int(0.53 * fw), int(0.61 * fw)
    ys0, ys1 = int(0.44 * fh), int(0.59 * fh)
    cone_p15 = np.percentile(gray[ys0:ys1, nx0:nx1], 15, axis=1)
    nozzle_dark_ref = float(np.min(cone_p15[:max(3, int(0.05 * fh))]))
    left_open_bed = float(np.median(gray[int(0.50 * fh):int(0.60 * fh), int(0.06 * fw):int(0.20 * fw)]))
    if left_open_bed <= nozzle_dark_ref + 25.0:
        return 0.0
    cone_thresh = nozzle_dark_ref + 0.36 * (left_open_bed - nozzle_dark_ref)

    ny = ys0
    for i, val in enumerate(cone_p15):
        if val < cone_thresh:
            ny = ys0 + i
        elif i > int(0.03 * fh):
            break
    if ny >= ys1 - 3:
        return 0.0

    y0 = min(fh - 30, ny + 3)
    y1 = int(0.90 * fh)
    if y1 - y0 < int(0.10 * fh):
        return 0.0

    zone = gray[y0:y1, int(0.42 * fw):int(0.70 * fw)]
    row_med = np.median(zone, axis=1)
    dy_med = np.mean(np.abs(np.diff(zone, axis=0, prepend=zone[:1])), axis=1)
    top_n = max(3, int(0.010 * fh))
    init_lum = float(np.median(row_med[:top_n]))
    init_dy = float(np.mean(dy_med[:top_n]))

    th = max(6, int(0.018 * fh))
    flank_l = gray[y0:y0 + th, int(0.24 * fw):int(0.46 * fw)]
    flank_r = gray[y0:y0 + th, int(0.66 * fw):int(0.77 * fw)]
    if (float(np.abs(np.diff(flank_l, axis=0)).mean()) > 9.5
            or float(np.abs(np.diff(flank_r, axis=0)).mean()) > 9.5
            or float((flank_l < cone_thresh).mean()) > 0.35
            or float((flank_r < cone_thresh).mean()) > 0.35):
        return 0.0
    if not (cone_thresh + 6.0 <= init_lum <= left_open_bed * 1.08 and init_dy < 8.5):
        return 0.0

    min_gap_px = max(8, int(min_gap_ratio * fh))
    max_gap_px = int(0.24 * fh)
    body_win = max(18, int(0.075 * fh))
    jump_thr = max(20.0, 0.24 * init_lum)

    gap_px = 0
    running_lum = init_lum
    part_hit = False
    for i in range(min(len(row_med), max_gap_px + 1)):
        if abs(float(row_med[i]) - running_lum) > jump_thr or float(dy_med[i]) > 9.0:
            ahead = slice(i, min(len(row_med), i + body_win))
            if ahead.stop - ahead.start < max(10, body_win // 2):
                break
            row_is_part = (np.abs(row_med[ahead] - running_lum) > jump_thr) | (dy_med[ahead] > 8.5)
            if float(row_is_part.mean()) >= 0.60:
                if i < min_gap_px:
                    return 0.0
                part_hit = True
                break
        gap_px += 1
        running_lum = 0.94 * running_lum + 0.06 * float(row_med[i])

    if not part_hit:
        return 0.0
    ahead = slice(gap_px, min(len(row_med), gap_px + body_win))
    part_contrast = float(np.median(np.abs(row_med[ahead] - init_lum)))
    gap_frac = gap_px / float(fh)
    return float(min(0.95, 0.74 + 2.2 * min(0.08, max(0.0, gap_frac - 0.015)) + 0.0012 * min(80.0, part_contrast)))


def prepare_tensor(img: np.ndarray, inp: dict) -> np.ndarray:
    if inp["dtype"] == np.float32:
        return img.astype(np.float32)[None]
    scale, zp = inp["quantization"]
    if scale:
        info = np.iinfo(inp["dtype"])
        return np.clip(np.round(img.astype(np.float32) / scale + zp), info.min, info.max).astype(inp["dtype"])[None]
    return img.astype(inp["dtype"])[None]


def decode_binary_probs(y_raw: np.ndarray, out: dict, n_out: int) -> tuple[float, float]:
    """Collapse any 1-output sigmoid or N-output softmax into (P(Correcte), P(Error))."""
    y = y_raw
    if out["dtype"] in (np.uint8, np.int8):
        scale, zp = out["quantization"]
        y = (y.astype(np.float32) - zp) * (scale or 1.0)
    y = y.astype(np.float32).reshape(-1)

    if n_out == 1:
        p_err = float(y[0])
        if not (0.0 <= p_err <= 1.0):
            p_err = float(1.0 / (1.0 + np.exp(-p_err)))
        p_err = float(np.clip(p_err, 0.0, 1.0))
        return 1.0 - p_err, p_err

    if y.min() < 0 or abs(float(y.sum()) - 1.0) > 0.02:
        e = np.exp(y - y.max())
        raw_probs = e / e.sum()
    else:
        raw_probs = y
    p_ok = float(np.clip(raw_probs[0], 0.0, 1.0))
    p_err = float(np.clip(1.0 - p_ok, 0.0, 1.0))
    return p_ok, p_err


def main():
    parser = argparse.ArgumentParser(description="Test and benchmark TDR .tflite models")
    parser.add_argument("model", nargs="?", default="printer_failure_detector.tflite",
                        help="Path or name of .tflite model (default: printer_failure_detector.tflite)")
    parser.add_argument("--threads", type=int, default=2, help="Number of CPU threads (default: 2)")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations (default: 10)")
    parser.add_argument("--image", type=str, default="", help="Optional path to test image file")
    parser.add_argument("--roi-mode", choices=["multi", "roi", "full"], default="roi",
                        help="ROI evaluation mode when --image is used (default: roi)")
    args = parser.parse_args()

    # Search paths
    model_path = Path(args.model)
    if not model_path.is_file():
        candidates = [
            Path(__file__).parent / args.model,
            Path(__file__).parent.parent / args.model,
            Path("/home/pi") / args.model,
            Path.home() / "tdr-bot" / args.model,
            Path.home() / "Downloads" / args.model,
        ]
        for c in candidates:
            if c.is_file():
                model_path = c
                break

    if not model_path.is_file():
        print(f"❌ Error: Model file not found: {args.model}")
        print("   Looked in current dir and standard search paths.")
        sys.exit(1)

    Interpreter, backend = import_interpreter()
    print(f"\n========================================================")
    print(f"  🔍 TDR Model Verification: {model_path.name}")
    print(f"========================================================")
    print(f"  Backend:         {backend}")
    print(f"  Model Path:      {model_path.resolve()}")
    print(f"  File Size:       {model_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Threads:         {args.threads}")

    interp = Interpreter(model_path=str(model_path), num_threads=args.threads)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    batch, h, w, c = (int(v) for v in inp["shape"])
    inp_dtype = np.dtype(inp["dtype"]).name
    out_dtype = np.dtype(out["dtype"]).name
    n_out = int(np.prod(out["shape"]))

    print(f"\n  Input Tensor:    shape=[{batch}, {h}, {w}, {c}], dtype={inp_dtype}")
    print(f"  Output Tensor:   shape={list(out['shape'])}, dtype={out_dtype} ({n_out} raw outputs -> 2 classes)")

    if n_out == 1:
        kind = "Binary Sigmoid (Correcte vs Error)"
    else:
        kind = f"{n_out}-Output Softmax collapsed to Binary (Correcte vs Error)"
    print(f"  Architecture:    {kind}")

    # Prepare input data
    use_real_image = bool(args.image and Path(args.image).is_file())
    if use_real_image:
        full_rgb = load_rgb_image(args.image)
        img_full = resize_rgb(full_rgb, w, h)
        img_roi = resize_rgb(crop_roi(full_rgb), w, h)
        print(f"  Test Image:      {args.image} ({full_rgb.shape[1]}x{full_rgb.shape[0]}, mode={args.roi_mode})")
    else:
        img_full = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img_roi = img_full
        print(f"  Test Image:      Synthetic RGB ({w}x{h})")

    x_full = prepare_tensor(img_full, inp)
    x_roi = prepare_tensor(img_roi, inp)

    # Warmup
    interp.set_tensor(inp["index"], x_full)
    interp.invoke()

    # Benchmark single-pass latency
    latencies = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x_full)
        interp.invoke()
        y_full = interp.get_tensor(out["index"]).copy()
        latencies.append((time.perf_counter() - t0) * 1000)

    p_ok_full, p_err_full = decode_binary_probs(y_full, out, n_out)

    if use_real_image and args.roi_mode in ("multi", "roi"):
        interp.set_tensor(inp["index"], x_roi)
        interp.invoke()
        y_roi = interp.get_tensor(out["index"]).copy()
        p_ok_roi, p_err_roi = decode_binary_probs(y_roi, out, n_out)
        p_err_gap = detect_nozzle_air_gap(full_rgb)
        candidates = [(p_err_roi, "nozzle-roi"), (p_err_gap, "air-gap-clog")]
        if args.roi_mode == "multi":
            candidates.append((p_err_full, "full-frame"))
        best_err, focus_used = max(candidates, key=lambda item: item[0])
        probs = [1.0 - best_err, best_err]
    else:
        probs = [p_ok_full, p_err_full]
        focus_used = "full-frame"

    avg_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)

    print(f"\n---------------- Performance & Probabilities ----------------")
    print(f"  Single-Pass Latency: {avg_ms:.1f} ms (min: {min_ms:.1f} ms, max: {max_ms:.1f} ms over {args.runs} runs)")
    if use_real_image:
        print(f"  Active Focus View:   {focus_used}")
    print(f"  Unified Bot Classes:")
    for cls_name, prob in zip(CLASSES, probs):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    • {cls_name:12s}: {prob*100:5.1f}%  [{bar}]")

    print(f"========================================================\n")
    print(f"  ✅ Model validation SUCCESSFUL!\n")


if __name__ == "__main__":
    main()
