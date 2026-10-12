"""YOLO-based object detection for COLLAB Video Lab.

The Ultralytics import is intentionally lazy so COLLAB can start even when the
optional YOLO dependency has not been installed yet.
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

_ALLOWED_MODELS = {"yolo26n.pt", "yolo26s.pt"}
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_model(model_name: str):
    if model_name not in _ALLOWED_MODELS:
        raise ValueError("Choose one of the supported models: yolo26n.pt or yolo26s.pt.")
    with _MODEL_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO is not installed in this environment. Run "
                "'.venv\\Scripts\\python.exe -m pip install ultralytics imageio-ffmpeg' "
                "from the COLLAB project folder, then restart Flask."
            ) from error
        try:
            # The first run downloads the model weights. Subsequent runs use the local cache.
            model = YOLO(model_name)
        except Exception as error:
            raise RuntimeError(
                f"Could not load {model_name}. On the first run, connect to the internet "
                "so Ultralytics can download the model weights, or put the weights in "
                "the local Ultralytics model cache. Details: " + str(error)
            ) from error
        _MODEL_CACHE[model_name] = model
        return model


def _names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def detect_video(
    input_path: str | Path,
    output_path: str | Path,
    csv_path: str | Path,
    *,
    confidence: float = 0.25,
    image_size: int = 640,
    model_name: str = "yolo26n.pt",
    class_filter: str = "",
) -> dict[str, Any]:
    """Annotate every decoded frame and write per-frame detections to CSV.

    `class_counts` counts detections across frames. It is not a count of unique
    tracked objects; persistent IDs require a tracking stage.
    """
    input_path, output_path, csv_path = Path(input_path), Path(output_path), Path(csv_path)
    if not input_path.is_file():
        raise FileNotFoundError("Source video was not found. Upload it again and retry.")
    confidence = float(confidence)
    if not 0.05 <= confidence <= 0.95:
        raise ValueError("Confidence threshold must be between 0.05 and 0.95.")
    if image_size not in {320, 480, 640, 960}:
        raise ValueError("Image size must be 320, 480, 640, or 960.")

    model = _load_model(model_name)
    names = _names_map(getattr(model, "names", {}))
    requested_names = {name.strip().casefold() for name in class_filter.split(",") if name.strip()}
    if requested_names:
        by_name = {name.casefold(): class_id for class_id, name in names.items()}
        unknown = sorted(name for name in requested_names if name not in by_name)
        if unknown:
            available_examples = ", ".join(sorted(set(names.values()))[:35])
            raise ValueError(
                "Unknown class filter: " + ", ".join(unknown) + ". Model classes include: "
                + available_examples
                + ". Use class names from the model."
            )
        class_ids = sorted(by_name[name] for name in requested_names)
    else:
        class_ids = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    h264_path = output_path.with_name(output_path.stem + ".h264.partial.mp4")
    for path in (raw_path, h264_path):
        path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    writer = None
    frame_count = 0
    detection_total = 0
    class_counts: Counter[str] = Counter()
    started = time.monotonic()
    csv_tmp = csv_path.with_suffix(".csv.tmp")

    if not cap.isOpened():
        cap.release()
        raise ValueError("OpenCV could not decode the uploaded source video.")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= fps <= 120.0:
            fps = 25.0
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Could not determine source video dimensions.")
        # H.264/yuv420p expects even dimensions. Pad only the bottom/right edge if needed.
        width = source_width + (source_width % 2)
        height = source_height + (source_height % 2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height), True)
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the detection output video.")

        with csv_tmp.open("w", newline="", encoding="utf-8-sig") as csv_file:
            fieldnames = [
                "frame_index", "timestamp_seconds", "class_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result_list = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    classes=class_ids,
                    verbose=False,
                )
                result = result_list[0] if result_list else None
                annotated = result.plot() if result is not None else frame.copy()
                boxes = getattr(result, "boxes", None) if result is not None else None

                if boxes is not None and len(boxes) > 0:
                    coords = boxes.xyxy.cpu().numpy()
                    class_values = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()
                    for box, class_id, score in zip(coords, class_values, confidences):
                        class_name = names.get(int(class_id), str(int(class_id)))
                        if requested_names and class_name.casefold() not in requested_names:
                            continue
                        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                        csv_writer.writerow({
                            "frame_index": frame_count,
                            "timestamp_seconds": round(frame_count / fps, 3),
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "confidence": round(float(score), 5),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        })
                        detection_total += 1
                        class_counts[class_name] += 1

                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(annotated)
                frame_count += 1

        writer.release()
        writer = None
        cap.release()
        cap = None
        if frame_count == 0:
            raise ValueError("No video frames could be decoded for object detection.")

        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "Browser-compatible detection video requires imageio-ffmpeg. "
                "Install it with: python -m pip install imageio-ffmpeg"
            ) from error
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-nostdin", "-y", "-i", str(raw_path), "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(h264_path),
        ]
        encoded = subprocess.run(command, capture_output=True, text=True, check=False)
        if encoded.returncode != 0 or not h264_path.is_file() or h264_path.stat().st_size < 512:
            detail = (encoded.stderr or "").strip()
            raise RuntimeError("Failed to create browser-compatible H.264 detection video. " + detail[-1000:])

        check = cv2.VideoCapture(str(h264_path))
        try:
            first_ok, first_frame = check.read()
            if not check.isOpened() or not first_ok or first_frame is None:
                raise RuntimeError("The detection output could not be decoded after H.264 encoding.")
        finally:
            check.release()

        os.replace(h264_path, output_path)
        os.replace(csv_tmp, csv_path)
        return {
            "frames_processed": frame_count,
            "detections_total": detection_total,
            "class_counts": dict(sorted(class_counts.items(), key=lambda item: item[0].casefold())),
            "fps": round(fps, 3),
            "width": source_width,
            "height": source_height,
            "output_size_bytes": output_path.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "model": model_name,
            "confidence": confidence,
            "image_size": image_size,
        }
    except Exception:
        output_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)
        h264_path.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
