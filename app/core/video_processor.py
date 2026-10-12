"""COLLAB video-processing pipeline.

Writes to a temporary MP4 first and publishes processed.mp4 only after the
writer has been released and the generated file has been checked. The FFT
video operation is implemented locally to keep it independent of image-workspace
code paths.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

SUPPORTED_SPEEDS = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Return a grayscale view/copy for BGR, BGRA or grayscale input."""
    if frame is None:
        raise ValueError("Cannot convert an empty frame to grayscale.")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _normalize_uint8(image: Any) -> np.ndarray:
    """Convert an image/result to a finite uint8 array for video encoding."""
    if image is None:
        raise ValueError("The image-processing algorithm returned no frame.")

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("The processed frame is empty.")

    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    if arr.ndim == 3 and arr.shape[2] == 2:
        real = arr[:, :, 0].astype(np.float32)
        imag = arr[:, :, 1].astype(np.float32)
        arr = np.sqrt(real * real + imag * imag)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr = arr.astype(np.float32, copy=False)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _prepare_video_frame(result: Any, width: int, height: int) -> np.ndarray:
    """Ensure every result is a three-channel BGR frame with fixed dimensions."""
    frame = _normalize_uint8(result)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3:
        if frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            raise ValueError(f"Unsupported video frame channels: {frame.shape[2]}")
    else:
        raise ValueError(f"Unsupported processed frame shape: {frame.shape}")

    if frame.shape[:2] != (height, width):
        interpolation = (
            cv2.INTER_AREA
            if frame.shape[1] > width or frame.shape[0] > height
            else cv2.INTER_CUBIC
        )
        frame = cv2.resize(frame, (width, height), interpolation=interpolation)

    return np.ascontiguousarray(frame)


def _fft_spectrum(frame: np.ndarray) -> np.ndarray:
    """Create a displayable log-magnitude FFT spectrum for one video frame."""
    gray = _to_gray(frame).astype(np.float32, copy=False)

    # OpenCV's DFT keeps the calculation within OpenCV and avoids passing any
    # complex-valued result into the general image pipeline.
    complex_spectrum = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    complex_spectrum = np.fft.fftshift(complex_spectrum, axes=(0, 1))
    magnitude = cv2.magnitude(complex_spectrum[:, :, 0], complex_spectrum[:, :, 1])
    magnitude = np.log1p(magnitude)
    return _normalize_uint8(magnitude)


def temporal_correlation_cancel(
    current_frame: np.ndarray,
    previous_frame: np.ndarray | None,
    current_weight: float = 0.5,
) -> np.ndarray:
    """Align and blend an adjacent frame to reduce temporal noise.

    The blend is skipped when dimensions differ or a large scene change is
    detected, reducing ghosting at cuts.
    """
    if current_frame is None:
        return current_frame
    if previous_frame is None or current_frame.shape != previous_frame.shape:
        return current_frame.copy()

    current_gray = _to_gray(current_frame)
    previous_gray = _to_gray(previous_frame)
    difference = float(np.mean(cv2.absdiff(current_gray, previous_gray)))
    if difference > 45.0:
        return current_frame.copy()

    flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0,
    )
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[:, :, 0]
    map_y = grid_y + flow[:, :, 1]
    aligned_previous = cv2.remap(
        previous_frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    current_weight = float(np.clip(current_weight, 0.0, 1.0))
    return cv2.addWeighted(
        current_frame, current_weight,
        aligned_previous, 1.0 - current_weight,
        0,
    )


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    operation: str,
    apply_operation_fn: Callable[[np.ndarray, str, dict], Any],
    parameters: dict | None = None,
    temporal_cancel: bool = False,
    force_grayscale: bool = False,
    playback_speed: float = 1.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Process a video frame-by-frame and publish a valid MP4 on success.

    Supported speed choices control the encoded frame rate: 0.5 means half
    speed (approximately twice the duration), while 2.0 means double speed.
    The callback receives (frames_processed, reported_total_frames).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    parameters = dict(parameters or {})
    operation = str(operation or "original").strip().lower()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    try:
        playback_speed = float(playback_speed)
    except (TypeError, ValueError) as error:
        raise ValueError("Playback speed must be numeric.") from error
    if playback_speed not in SUPPORTED_SPEEDS:
        raise ValueError(f"Unsupported playback speed: {playback_speed}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A separate filename prevents browsers from serving a half-written MP4.
    temporary_path = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)
    browser_path = output_path.with_name(
        f"{output_path.stem}.h264.partial{output_path.suffix}"
    )
    browser_path.unlink(missing_ok=True)
    # Do not leave an older processed file visible if this new attempt fails.
    output_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    writer = None
    processed_frames = 0
    started_at = time.monotonic()
    source_fps = 25.0
    total_frames = 0
    width = height = 0
    released_writer = False

    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open the input video or its codec.")

    try:
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if np.isfinite(reported_fps) and 1.0 <= reported_fps <= 120.0:
            source_fps = reported_fps
        output_fps = source_fps * playback_speed

        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Could not determine input video dimensions.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temporary_path), fourcc, output_fps, (width, height), True
        )
        if not writer.isOpened():
            raise RuntimeError(
                "Could not create the temporary output video. Check OpenCV codec support."
            )

        previous_base_frame = None
        print(
            f"[COLLAB video] Start: operation={operation}, "
            f"frames={total_frames or 'unknown'}, size={width}x{height}, "
            f"fps={source_fps:.3f}, speed={playback_speed}x",
            flush=True,
        )

        while True:
            read_ok, source_frame = capture.read()
            if not read_ok:
                break

            current_source_frame = source_frame.copy()
            if force_grayscale:
                current_base_frame = cv2.cvtColor(
                    current_source_frame, cv2.COLOR_BGR2GRAY
                )
            else:
                current_base_frame = current_source_frame.copy()

            if temporal_cancel and previous_base_frame is not None:
                current_base_frame = temporal_correlation_cancel(
                    current_base_frame,
                    previous_base_frame,
                    current_weight=0.5,
                )

            # Use a dedicated, bounded implementation for video FFT instead of
            # relying on any image-workspace-specific FFT return type.
            if operation == "fft":
                result = _fft_spectrum(current_base_frame)
            else:
                result = apply_operation_fn(
                    current_base_frame, operation, parameters
                )

            output_frame = _prepare_video_frame(result, width, height)
            writer.write(output_frame)
            processed_frames += 1
            previous_base_frame = current_source_frame

            if progress_callback is not None and (
                processed_frames % 5 == 0
                or (total_frames > 0 and processed_frames == total_frames)
            ):
                progress_callback(processed_frames, total_frames)

            if processed_frames % 25 == 0:
                print(
                    f"[COLLAB video] {processed_frames}/"
                    f"{total_frames or '?'} frames processed",
                    flush=True,
                )

        if processed_frames == 0:
            raise ValueError("No video frames could be decoded.")

        # Finalize the MP4 container before attempting validation or publication.
        writer.release()
        writer = None
        released_writer = True
        capture.release()
        capture = None

        check = cv2.VideoCapture(str(temporary_path))
        try:
            if not check.isOpened():
                raise RuntimeError(
                    "OpenCV could not reopen the completed temporary MP4. "
                    "The output was not published."
                )
            first_frame_ok, first_frame = check.read()
            reported_output_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            if not first_frame_ok or first_frame is None:
                raise RuntimeError(
                    "The output MP4 was written but its frames cannot be decoded."
                )
        finally:
            check.release()

        # OpenCV's mp4v output is playable in many desktop media players but
        # is not supported by every browser's HTML5 video decoder. Re-encode
        # to H.264/yuv420p and move the MP4 index to the front for browser seek.
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "Browser-compatible video export requires imageio-ffmpeg. "
                "Install it in the COLLAB virtual environment with: "
                "python -m pip install imageio-ffmpeg, then process the video again."
            ) from error

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-i", str(temporary_path),
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-f", "mp4",
            str(browser_path),
        ]
        transcoded = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if transcoded.returncode != 0:
            detail = (transcoded.stderr or "").strip()
            raise RuntimeError(
                "Could not encode a browser-compatible H.264 MP4. "
                + (detail[-1200:] if detail else "FFmpeg returned an error.")
            )
        if not browser_path.is_file() or browser_path.stat().st_size < 512:
            raise RuntimeError("FFmpeg did not produce a valid H.264 MP4 file.")

        # Validate the exact H.264 file that the browser will receive.
        browser_check = cv2.VideoCapture(str(browser_path))
        try:
            browser_frame_ok, browser_frame = browser_check.read()
            if not browser_check.isOpened() or not browser_frame_ok or browser_frame is None:
                raise RuntimeError(
                    "The H.264 export was created but OpenCV could not decode its first frame."
                )
        finally:
            browser_check.release()

        os.replace(browser_path, output_path)
        temporary_path.unlink(missing_ok=True)
        elapsed = time.monotonic() - started_at
        print(
            f"[COLLAB video] Completed: {processed_frames} frames in "
            f"{elapsed:.1f}s; browser-compatible H.264 output={output_path}",
            flush=True,
        )

        return {
            "frames_processed": processed_frames,
            "total_frames_reported": total_frames,
            "output_frames_reported": reported_output_frames,
            "fps": output_fps,
            "source_fps": source_fps,
            "playback_speed": playback_speed,
            "width": width,
            "height": height,
            "elapsed_seconds": round(elapsed, 2),
            "output_size_bytes": output_path.stat().st_size,
        }

    except Exception:
        # A partial container is never left behind under the final output name.
        temporary_path.unlink(missing_ok=True)
        browser_path.unlink(missing_ok=True)
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None and not released_writer:
            writer.release()
        # If the run failed, remove the unfinished temporary output.
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        if browser_path.exists():
            browser_path.unlink(missing_ok=True)
