"""Local OCR for COLLAB Video Lab using Tesseract OCR.

Processes a source video, runs OCR on every Nth frame, draws recognized text
boxes on sampled frames, and exports an H.264 video plus CSV/TXT results.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2


def _get_pytesseract():
    """Import pytesseract and resolve common Windows Tesseract locations."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as error:
        raise RuntimeError(
            "The Python OCR package is missing. In COLLAB's activated .venv, run: "
            "python -m pip install pytesseract"
        ) from error

    configured = os.environ.get("TESSERACT_CMD", "").strip()
    candidates = [configured] if configured else []
    located = shutil.which("tesseract")
    if located:
        candidates.append(located)
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates.extend([
        str(Path(program_files) / "Tesseract-OCR" / "tesseract.exe"),
        str(Path(local_app_data) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
        if local_app_data else "",
    ])

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        raise RuntimeError(
            "Tesseract OCR engine was not found. Install Tesseract OCR for Windows, "
            "then restart COLLAB. If it is installed in a custom folder, set the "
            "TESSERACT_CMD environment variable to the full path of tesseract.exe."
        ) from error
    return pytesseract, Output


def _ocr_lines(frame, pytesseract, output_type, *, language: str, min_confidence: float):
    """Return line-level text records and their boxes in original-frame pixels."""
    height, width = frame.shape[:2]
    scale = 1.5 if max(width, height) < 1600 else 1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Sparse-text mode is useful for signs, labels, screens, and scene text.
    data = pytesseract.image_to_data(
        gray,
        lang=language,
        config="--oem 3 --psm 11",
        output_type=output_type.DICT,
        timeout=30,
    )

    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    count = len(data.get("text", []))
    for i in range(count):
        word = str(data["text"][i]).strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        left = int(round(int(data["left"][i]) / scale))
        top = int(round(int(data["top"][i]) / scale))
        box_width = int(round(int(data["width"][i]) / scale))
        box_height = int(round(int(data["height"][i]) / scale))
        grouped.setdefault(key, []).append({
            "text": word,
            "confidence": confidence,
            "x1": max(0, left),
            "y1": max(0, top),
            "x2": min(width - 1, left + max(1, box_width)),
            "y2": min(height - 1, top + max(1, box_height)),
            "left_order": left,
        })

    lines = []
    for words in grouped.values():
        words.sort(key=lambda item: item["left_order"])
        text = " ".join(item["text"] for item in words).strip()
        if not text:
            continue
        lines.append({
            "text": text,
            "confidence": sum(item["confidence"] for item in words) / len(words),
            "x1": min(item["x1"] for item in words),
            "y1": min(item["y1"] for item in words),
            "x2": max(item["x2"] for item in words),
            "y2": max(item["y2"] for item in words),
        })
    lines.sort(key=lambda item: (item["y1"], item["x1"]))
    return lines


def _draw_ocr_boxes(frame, lines):
    """Draw readable boxes and labels without mutating the source frame."""
    annotated = frame.copy()
    for item in lines:
        x1, y1, x2, y2 = (int(item[k]) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 120), 2)
        label = item["text"].replace("\n", " ").strip()
        if len(label) > 70:
            label = label[:67] + "..."
        label_y = max(20, y1 - 7)
        cv2.putText(
            annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (40, 220, 120), 2, cv2.LINE_AA,
        )
    return annotated


def extract_video_text(
    input_path: str | Path,
    output_path: str | Path,
    csv_path: str | Path,
    text_path: str | Path,
    *,
    language: str = "eng",
    frame_stride: int = 5,
    min_confidence: float = 35.0,
) -> dict[str, Any]:
    """OCR a video and export annotated MP4, line-level CSV, and timestamped TXT.

    OCR runs on frame indexes 0, frame_stride, 2*frame_stride, ...; the output
    video retains every frame, with text boxes drawn on sampled OCR frames.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    csv_path = Path(csv_path)
    text_path = Path(text_path)

    if not input_path.is_file():
        raise FileNotFoundError("Source video was not found. Upload it again and retry.")
    if language not in {"eng", "hin", "eng+hin"}:
        raise ValueError("Choose a supported OCR language: eng, hin, or eng+hin.")
    if not 1 <= int(frame_stride) <= 30:
        raise ValueError("Frame sampling interval must be from 1 to 30.")
    if not 0 <= float(min_confidence) <= 100:
        raise ValueError("OCR confidence must be between 0 and 100.")

    pytesseract, output_type = _get_pytesseract()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    h264_path = output_path.with_name(output_path.stem + ".h264.partial.mp4")
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    text_tmp = text_path.with_suffix(".txt.tmp")
    for path in (output_path, raw_path, h264_path, csv_path, text_path, csv_tmp, text_tmp):
        path.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    writer = None
    started = time.monotonic()
    frames_written = 0
    sampled_frames = 0
    lines_total = 0
    fps = 25.0

    if not cap.isOpened():
        cap.release()
        raise ValueError("OpenCV could not decode the uploaded source video.")

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if 1.0 <= source_fps <= 120.0:
            fps = source_fps
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Could not determine source video dimensions.")

        width = source_width + (source_width % 2)
        height = source_height + (source_height % 2)
        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the OCR preview video.")

        with csv_tmp.open("w", newline="", encoding="utf-8-sig") as csv_file, \
             text_tmp.open("w", encoding="utf-8") as text_file:
            fields = [
                "frame_index", "timestamp_seconds", "timestamp_hhmmss", "text",
                "confidence", "x1", "y1", "x2", "y2",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                output_frame = frame
                if frames_written % int(frame_stride) == 0:
                    sampled_frames += 1
                    lines = _ocr_lines(
                        frame, pytesseract, output_type,
                        language=language, min_confidence=float(min_confidence),
                    )
                    if lines:
                        output_frame = _draw_ocr_boxes(frame, lines)
                    timestamp = frames_written / fps
                    whole_seconds = int(timestamp)
                    timestamp_label = (
                        f"{whole_seconds // 3600:02d}:"
                        f"{(whole_seconds % 3600) // 60:02d}:"
                        f"{timestamp % 60:05.2f}"
                    )
                    for item in lines:
                        record = {
                            "frame_index": frames_written,
                            "timestamp_seconds": round(timestamp, 3),
                            "timestamp_hhmmss": timestamp_label,
                            "text": item["text"],
                            "confidence": round(float(item["confidence"]), 2),
                            "x1": item["x1"], "y1": item["y1"],
                            "x2": item["x2"], "y2": item["y2"],
                        }
                        csv_writer.writerow(record)
                        text_file.write(f"[{timestamp_label}] {item['text']}\n")
                        lines_total += 1

                if output_frame.shape[1] != width or output_frame.shape[0] != height:
                    output_frame = cv2.resize(output_frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(output_frame)
                frames_written += 1

        writer.release()
        writer = None
        cap.release()
        cap = None
        if frames_written == 0:
            raise ValueError("No video frames could be decoded for OCR.")

        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "H.264 export requires imageio-ffmpeg. Install it with: "
                "python -m pip install imageio-ffmpeg"
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
            raise RuntimeError("Failed to create browser-compatible H.264 OCR video. " + detail[-1000:])

        check = cv2.VideoCapture(str(h264_path))
        try:
            first_ok, first_frame = check.read()
            if not check.isOpened() or not first_ok or first_frame is None:
                raise RuntimeError("The OCR output could not be decoded after H.264 encoding.")
        finally:
            check.release()

        os.replace(h264_path, output_path)
        os.replace(csv_tmp, csv_path)
        os.replace(text_tmp, text_path)
        return {
            "frames_processed": frames_written,
            "frames_ocr_scanned": sampled_frames,
            "text_lines_found": lines_total,
            "fps": round(fps, 3),
            "width": source_width,
            "height": source_height,
            "output_size_bytes": output_path.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "language": language,
            "frame_stride": int(frame_stride),
            "min_confidence": float(min_confidence),
        }
    except Exception:
        output_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        text_path.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        text_tmp.unlink(missing_ok=True)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)
        h264_path.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        text_tmp.unlink(missing_ok=True)
