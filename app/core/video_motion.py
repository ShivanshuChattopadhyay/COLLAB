"""COLLAB motion estimation and motion-compensated prediction.

This module is a separate opt-in analysis path. It does not change the regular
Video Lab frame-processing pipeline, so existing filters do not incur the cost
of block matching unless the user starts this analysis explicitly.

Outputs:
- motion_analysis.mp4: four-panel visualisation (current, predicted, residual,
  and displacement vectors)
- motion_vectors.csv: per-frame/per-block displacement vectors and match costs
- motion_metrics.csv: naive previous-frame MAE vs compensated-prediction MAE

The block-matching mode uses OpenCV's efficient TM_SQDIFF template matching
(SSD-style cost), over downscaled grayscale frames. Global mode uses phase
correlation and estimates one translation vector per frame pair.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PRESETS = {
    # scale, block size in analysis pixels, search radius in analysis pixels
    "fast": (0.25, 24, 4),
    "balanced": (0.25, 16, 7),
    "accurate": (0.50, 16, 8),
}

VECTOR_FIELDS = [
    "frame_index", "timestamp_s", "vector_type", "block_x_px", "block_y_px",
    "dx_px", "dy_px", "matching_cost", "phase_response", "scene_cut", "algorithm",
]
METRIC_FIELDS = [
    "frame_index", "timestamp_s", "naive_previous_frame_mae",
    "motion_compensated_prediction_mae", "improvement_percent", "scene_cut",
]


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _scene_cut_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Compare coarse grayscale thumbnails to avoid mistaking fine motion for a cut."""
    prev_gray = _gray(previous)
    curr_gray = _gray(current)
    h, w = prev_gray.shape[:2]
    thumb_w, thumb_h = min(96, w), min(54, h)
    prev_thumb = cv2.resize(prev_gray, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    curr_thumb = cv2.resize(curr_gray, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(prev_thumb, curr_thumb)))


def _even(value: int) -> int:
    return max(2, int(value) // 2 * 2)


def _small_gray(frame: np.ndarray, scale: float) -> np.ndarray:
    height, width = frame.shape[:2]
    target_w = max(32, int(round(width * scale)))
    target_h = max(32, int(round(height * scale)))
    return cv2.resize(_gray(frame), (target_w, target_h), interpolation=cv2.INTER_AREA)


def _block_match_prediction(
    previous: np.ndarray,
    current: np.ndarray,
    scale: float,
    block_size: int,
    search_radius: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Estimate local motion using template matching and compensate by blocks."""
    prev_small = _small_gray(previous, scale)
    curr_small = _small_gray(current, scale)
    sh, sw = curr_small.shape[:2]
    full_h, full_w = current.shape[:2]
    sx = full_w / float(sw)
    sy = full_h / float(sh)

    # Keep unmatchable borders/occlusions anchored to the current frame rather
    # than injecting a bad copy from outside the reference image.
    prediction = current.copy()
    vectors: list[dict[str, Any]] = []
    block_size = max(8, int(block_size))
    search_radius = max(1, int(search_radius))

    for y in range(0, sh, block_size):
        for x in range(0, sw, block_size):
            bh = min(block_size, sh - y)
            bw = min(block_size, sw - x)
            if bh < 4 or bw < 4:
                continue
            template = curr_small[y:y + bh, x:x + bw]
            sx0 = max(0, x - search_radius)
            sy0 = max(0, y - search_radius)
            sx1 = min(sw, x + search_radius + bw + 1)
            sy1 = min(sh, y + search_radius + bh + 1)
            search = prev_small[sy0:sy1, sx0:sx1]
            if search.shape[0] < bh or search.shape[1] < bw:
                continue

            # TM_SQDIFF returns the minimum squared-difference matching cost.
            response = cv2.matchTemplate(search, template, cv2.TM_SQDIFF)
            min_cost, _, min_loc, _ = cv2.minMaxLoc(response)
            match_x = sx0 + int(min_loc[0])
            match_y = sy0 + int(min_loc[1])

            # Motion vector convention: current block location minus reference
            # block location. Compensation samples previous at (x - dx, y - dy).
            dx_small = x - match_x
            dy_small = y - match_y
            dx_full = int(round(dx_small * sx))
            dy_full = int(round(dy_small * sy))

            x0 = int(round(x * sx))
            x1 = min(full_w, int(round((x + bw) * sx)))
            y0 = int(round(y * sy))
            y1 = min(full_h, int(round((y + bh) * sy)))
            rx0, ry0 = x0 - dx_full, y0 - dy_full
            rx1, ry1 = rx0 + (x1 - x0), ry0 + (y1 - y0)

            if (
                x1 > x0 and y1 > y0 and rx0 >= 0 and ry0 >= 0
                and rx1 <= full_w and ry1 <= full_h
            ):
                patch = previous[ry0:ry1, rx0:rx1]
                if patch.shape == prediction[y0:y1, x0:x1].shape:
                    prediction[y0:y1, x0:x1] = patch

            vectors.append({
                "vector_type": "block",
                "block_x_px": int((x0 + x1) / 2),
                "block_y_px": int((y0 + y1) / 2),
                "dx_px": dx_full,
                "dy_px": dy_full,
                "matching_cost": round(float(min_cost) / float(bw * bh), 4),
                "phase_response": "",
            })
    return prediction, vectors


def _global_prediction(
    previous: np.ndarray, current: np.ndarray, scale: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fast global-translation estimate by phase correlation."""
    prev_small = _small_gray(previous, scale).astype(np.float32)
    curr_small = _small_gray(current, scale).astype(np.float32)
    shift, response = cv2.phaseCorrelate(prev_small, curr_small)
    sx = current.shape[1] / float(prev_small.shape[1])
    sy = current.shape[0] / float(prev_small.shape[0])
    # Very weak correlation is not a trustworthy global-motion estimate.
    if not math.isfinite(float(response)) or float(response) < 0.03:
        shift = (0.0, 0.0)
    dx = float(shift[0] * sx)
    dy = float(shift[1] * sy)
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    prediction = cv2.warpAffine(
        previous, matrix, (current.shape[1], current.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    vector = [{
        "vector_type": "global",
        "block_x_px": current.shape[1] // 2,
        "block_y_px": current.shape[0] // 2,
        "dx_px": round(dx, 3),
        "dy_px": round(dy, 3),
        "matching_cost": "",
        "phase_response": round(float(response), 5),
    }]
    return prediction, vector


def _draw_vectors(frame: np.ndarray, vectors: list[dict[str, Any]], algorithm: str) -> np.ndarray:
    canvas = frame.copy()
    if algorithm == "global":
        selected = vectors
    else:
        # Draw every second vector in each direction so dense grids remain legible.
        selected = vectors[::2]
    for item in selected:
        x = int(item["block_x_px"])
        y = int(item["block_y_px"])
        dx = int(item["dx_px"])
        dy = int(item["dy_px"])
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            end = (int(np.clip(x + dx, 0, canvas.shape[1] - 1)),
                   int(np.clip(y + dy, 0, canvas.shape[0] - 1)))
            cv2.arrowedLine(canvas, (x, y), end, (80, 230, 120), 1, tipLength=0.25)
            cv2.circle(canvas, (x, y), 1, (30, 150, 255), -1)
    return canvas


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 25), (15, 15, 18), -1)
    cv2.putText(out, label, (9, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _compose(current: np.ndarray, prediction: np.ndarray, residual: np.ndarray,
             vector_frame: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    def fit(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    input_panel = _label_panel(fit(current), "CURRENT FRAME")
    prediction_panel = _label_panel(fit(prediction), "MOTION-COMPENSATED PREDICTION")
    residual_panel = _label_panel(fit(residual), "ABSOLUTE PREDICTION RESIDUAL x3")
    vectors_panel = _label_panel(fit(vector_frame), "DISPLACEMENT VECTORS (dx, dy)")
    top = np.hstack((input_panel, prediction_panel))
    bottom = np.hstack((residual_panel, vectors_panel))
    return np.vstack((top, bottom))


def analyze_motion_video(
    input_path: str | Path,
    output_path: str | Path,
    vectors_csv_path: str | Path,
    metrics_csv_path: str | Path,
    algorithm: str = "block",
    preset: str = "fast",
    progress_callback=None,
) -> dict[str, Any]:
    """Create an H.264 motion-analysis video and CSV reports.

    `algorithm` is `block` (local block matching) or `global` (fast translation).
    `preset` trades speed against local search accuracy: fast, balanced, accurate.
    Processing is opt-in and happens only when this function is called.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    vectors_csv_path = Path(vectors_csv_path)
    metrics_csv_path = Path(metrics_csv_path)
    algorithm = str(algorithm).lower().strip()
    preset = str(preset).lower().strip()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if algorithm not in {"block", "global"}:
        raise ValueError("Choose block matching or global translation.")
    if preset not in PRESETS:
        raise ValueError("Choose the fast, balanced, or accurate preset.")

    scale, block_size, search_radius = PRESETS[preset]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    h264_path = output_path.with_name(output_path.stem + ".h264.partial.mp4")
    for path in (output_path, raw_path, h264_path, vectors_csv_path, metrics_csv_path):
        path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    writer = None
    started = time.monotonic()
    frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if capture.isOpened() else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    if not math.isfinite(fps) or not (1.0 <= fps <= 120.0):
        fps = 25.0
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open the source video or its codec.")

    frame_index = 0
    vector_rows_count = 0
    naive_errors: list[float] = []
    compensated_errors: list[float] = []
    total_h = total_w = 0
    panel_w = panel_h = 0
    prev_frame = None
    try:
        ok, first = capture.read()
        if not ok or first is None:
            raise ValueError("No frames could be decoded from the source video.")
        height, width = first.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Source video has invalid dimensions.")
        total_w, total_h = width, height
        display_scale = min(640.0 / width, 360.0 / height, 1.0)
        panel_w = _even(max(2, int(round(width * display_scale))))
        panel_h = _even(max(2, int(round(height * display_scale))))
        out_w, out_h = panel_w * 2, panel_h * 2
        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h), True
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the intermediate analysis video.")

        with vectors_csv_path.open("w", newline="", encoding="utf-8-sig") as vectors_file, \
             metrics_csv_path.open("w", newline="", encoding="utf-8-sig") as metrics_file:
            vector_writer = csv.DictWriter(vectors_file, fieldnames=VECTOR_FIELDS)
            metric_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
            vector_writer.writeheader()
            metric_writer.writeheader()

            while True:
                if frame_index == 0:
                    current = first
                else:
                    ok, current = capture.read()
                    if not ok or current is None:
                        break
                timestamp = frame_index / fps
                scene_cut = False
                vectors: list[dict[str, Any]] = []

                if prev_frame is None:
                    prediction = current.copy()
                    naive_mae = 0.0
                    compensated_mae = 0.0
                else:
                    prev_gray = _gray(prev_frame)
                    curr_gray = _gray(current)
                    naive_mae = float(np.mean(cv2.absdiff(curr_gray, prev_gray)))
                    # Use coarse thumbnails for cut detection: fine-grained texture
                    # motion can make full-resolution differences large even when
                    # the scene is continuous.
                    scene_cut = _scene_cut_score(prev_frame, current) > 45.0

                    if scene_cut:
                        # Treat a cut as a reference reset in the prediction stage.
                        # Keeping the old reference in this diagnostic makes the
                        # prediction failure visible in the residual panel/metrics.
                        prediction = prev_frame.copy()
                        vectors = [{
                            "vector_type": "scene_cut", "block_x_px": -1,
                            "block_y_px": -1, "dx_px": 0, "dy_px": 0,
                            "matching_cost": round(naive_mae, 4), "phase_response": "",
                        }]
                    elif algorithm == "global":
                        prediction, vectors = _global_prediction(prev_frame, current, scale)
                    else:
                        prediction, vectors = _block_match_prediction(
                            prev_frame, current, scale, block_size, search_radius
                        )

                    compensated_mae = float(
                        np.mean(cv2.absdiff(curr_gray, _gray(prediction)))
                    )

                denom = naive_mae
                improvement = ((naive_mae - compensated_mae) / denom * 100.0) if denom > 1e-9 else 0.0
                naive_errors.append(naive_mae)
                compensated_errors.append(compensated_mae)
                metric_writer.writerow({
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp, 4),
                    "naive_previous_frame_mae": round(naive_mae, 4),
                    "motion_compensated_prediction_mae": round(compensated_mae, 4),
                    "improvement_percent": round(improvement, 3),
                    "scene_cut": int(scene_cut),
                })

                for item in vectors:
                    vector_writer.writerow({
                        "frame_index": frame_index,
                        "timestamp_s": round(timestamp, 4),
                        "vector_type": item["vector_type"],
                        "block_x_px": item["block_x_px"],
                        "block_y_px": item["block_y_px"],
                        "dx_px": item["dx_px"],
                        "dy_px": item["dy_px"],
                        "matching_cost": item.get("matching_cost", ""),
                        "phase_response": item.get("phase_response", ""),
                        "scene_cut": int(scene_cut),
                        "algorithm": algorithm,
                    })
                    vector_rows_count += 1

                abs_error = cv2.absdiff(_gray(current), _gray(prediction))
                residual_vis = cv2.applyColorMap(cv2.convertScaleAbs(abs_error, alpha=3.0), cv2.COLORMAP_JET)
                vectors_frame = _draw_vectors(current, vectors, algorithm)
                composite = _compose(current, prediction, residual_vis, vectors_frame, panel_w, panel_h)
                writer.write(composite)
                frame_index += 1
                prev_frame = current.copy()

                if progress_callback is not None:
                    progress_callback(frame_index, frame_count)

        if frame_index == 0:
            raise ValueError("No video frames could be processed.")
        writer.release()
        writer = None
        capture.release()
        capture = None

        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "Browser-compatible export requires imageio-ffmpeg. Run "
                "python -m pip install imageio-ffmpeg in COLLAB's environment."
            ) from error

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(raw_path), "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4",
            str(h264_path),
        ]
        encoded = subprocess.run(command, capture_output=True, text=True, check=False)
        if encoded.returncode != 0:
            detail = (encoded.stderr or "").strip()
            raise RuntimeError("Could not encode H.264 analysis MP4. " + (detail[-1200:] if detail else "FFmpeg error."))
        if not h264_path.is_file() or h264_path.stat().st_size < 512:
            raise RuntimeError("FFmpeg did not create a valid motion-analysis MP4.")
        check = cv2.VideoCapture(str(h264_path))
        try:
            check_ok, check_frame = check.read()
            if not check.isOpened() or not check_ok or check_frame is None:
                raise RuntimeError("The exported H.264 motion-analysis MP4 could not be decoded.")
        finally:
            check.release()
        os.replace(h264_path, output_path)
        raw_path.unlink(missing_ok=True)

        elapsed = round(time.monotonic() - started, 2)
        return {
            "algorithm": "Block matching (TM_SQDIFF / SSD-style cost)" if algorithm == "block" else "Global translation (phase correlation)",
            "algorithm_key": algorithm,
            "preset": preset,
            "analysis_scale": scale,
            "block_size_analysis_pixels": block_size if algorithm == "block" else None,
            "search_radius_analysis_pixels": search_radius if algorithm == "block" else None,
            "frames_processed": frame_index,
            "total_frames_reported": frame_count,
            "source_fps": round(fps, 3),
            "width": total_w,
            "height": total_h,
            "vector_rows": vector_rows_count,
            "mean_naive_previous_frame_mae": round(float(np.mean(naive_errors)), 4),
            "mean_compensated_prediction_mae": round(float(np.mean(compensated_errors)), 4),
            "mean_error_improvement_percent": round(float(np.mean([
                ((a - b) / a * 100.0) if a > 1e-9 else 0.0
                for a, b in zip(naive_errors, compensated_errors)
            ])), 3),
            "elapsed_seconds": elapsed,
            "output_size_bytes": output_path.stat().st_size,
        }
    except Exception:
        output_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        h264_path.unlink(missing_ok=True)
        vectors_csv_path.unlink(missing_ok=True)
        metrics_csv_path.unlink(missing_ok=True)
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)
        h264_path.unlink(missing_ok=True)
