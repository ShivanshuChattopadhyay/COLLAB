"""Safely integrate COLLAB motion analysis into the existing local project.

Run from the COLLAB root after extracting this patch ZIP into that folder.
The script backs up app.py and templates/video.html before changing them.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
TEMPLATE = ROOT / "templates" / "video.html"
PROCESSOR = ROOT / "app" / "core" / "video_processor.py"
MODULE = ROOT / "app" / "core" / "video_motion.py"

ROUTES = r'''# ============================================================
# MOTION ESTIMATION, COMPENSATION AND PREDICTION-RESIDUAL ANALYSIS
# Kept separate from ordinary processing so block matching is opt-in.
# ============================================================

def write_motion_status(job_folder, **updates):
    """Persist motion-analysis state independently from regular processing."""
    path = job_folder / "motion_status.json"
    with _VIDEO_STATUS_LOCK:
        status_data = {}
        if path.exists():
            try:
                status_data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status_data = {}
        status_data.update(updates)
        status_data["updated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime()
        )
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_motion_status(job_folder):
    path = job_folder / "motion_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.route("/video/motion/<job_id>", methods=["POST"])
def video_motion_analysis(job_id):
    """Create motion vectors, a compensated prediction and residual diagnostics."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid video job."}), 400

    input_files = sorted(job_folder.glob("input.*"))
    if not input_files:
        return jsonify({"ok": False, "error": "Source video was not found. Upload it again."}), 404

    algorithm = request.form.get("algorithm", "block").strip().lower()
    preset = request.form.get("preset", "fast").strip().lower()
    if algorithm not in {"block", "global"}:
        return jsonify({"ok": False, "error": "Choose block matching or global translation."}), 400
    if preset not in {"fast", "balanced", "accurate"}:
        return jsonify({"ok": False, "error": "Choose the fast, balanced, or accurate preset."}), 400

    if not reserve_video_job(job_id):
        return jsonify({
            "ok": False,
            "error": "This video is already being processed. Wait for that operation to finish.",
        }), 409

    output_path = job_folder / "motion_analysis.mp4"
    vectors_path = job_folder / "motion_vectors.csv"
    metrics_path = job_folder / "motion_metrics.csv"
    for path in (output_path, vectors_path, metrics_path):
        path.unlink(missing_ok=True)

    try:
        write_motion_status(
            job_folder, state="processing", algorithm=algorithm, preset=preset,
            message="Estimating motion and building prediction-residual diagnostics.",
            error=None, summary=None,
        )
        # Import lazily so COLLAB's ordinary image/video tools remain usable if
        # motion analysis is not requested or its optional runtime dependency fails.
        from app.core.video_motion import analyze_motion_video
        result = analyze_motion_video(
            input_path=input_files[0],
            output_path=output_path,
            vectors_csv_path=vectors_path,
            metrics_csv_path=metrics_path,
            algorithm=algorithm,
            preset=preset,
        )
        video_url = url_for("video_motion_file", job_id=job_id)
        download_url = url_for("video_motion_download", job_id=job_id)
        vectors_url = url_for("video_motion_vectors", job_id=job_id)
        metrics_url = url_for("video_motion_metrics", job_id=job_id)
        write_motion_status(
            job_folder, state="completed", message="Motion analysis completed.",
            error=None, summary=result, output_url=video_url,
            download_url=download_url, vectors_url=vectors_url, metrics_url=metrics_url,
        )
        return jsonify({
            "ok": True, "message": "Motion analysis completed.",
            "video_url": video_url, "download_url": download_url,
            "vectors_url": vectors_url, "metrics_url": metrics_url,
            "summary": result,
        })
    except Exception as error:
        app.logger.exception("Motion analysis failed for %s", job_id)
        write_motion_status(
            job_folder, state="failed", message="Motion analysis failed.", error=str(error)
        )
        return jsonify({"ok": False, "error": str(error)}), 500
    finally:
        release_video_job(job_id)


@app.route("/video/motion/file/<job_id>")
def video_motion_file(job_id):
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    output_path = job_folder / "motion_analysis.mp4"
    if read_motion_status(job_folder).get("state") != "completed" or not output_path.is_file():
        return "Motion analysis output is not ready yet.", 404
    return send_file(output_path, mimetype="video/mp4", conditional=True)


@app.route("/video/motion/download/<job_id>")
def video_motion_download(job_id):
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    output_path = job_folder / "motion_analysis.mp4"
    if read_motion_status(job_folder).get("state") != "completed" or not output_path.is_file():
        return "Motion analysis output is not ready yet.", 404
    return send_file(
        output_path, as_attachment=True, download_name="collab_motion_analysis.mp4",
        mimetype="video/mp4", conditional=True,
    )


@app.route("/video/motion/vectors/<job_id>")
def video_motion_vectors(job_id):
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    csv_path = job_folder / "motion_vectors.csv"
    if read_motion_status(job_folder).get("state") != "completed" or not csv_path.is_file():
        return "Motion vectors are not ready yet.", 404
    return send_file(
        csv_path, as_attachment=True, download_name="collab_motion_vectors.csv",
        mimetype="text/csv; charset=utf-8", conditional=True,
    )


@app.route("/video/motion/metrics/<job_id>")
def video_motion_metrics(job_id):
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    csv_path = job_folder / "motion_metrics.csv"
    if read_motion_status(job_folder).get("state") != "completed" or not csv_path.is_file():
        return "Motion metrics are not ready yet.", 404
    return send_file(
        csv_path, as_attachment=True, download_name="collab_motion_metrics.csv",
        mimetype="text/csv; charset=utf-8", conditional=True,
    )


'''

FAST_TEMPORAL_FUNCTION = r'''def temporal_correlation_cancel(
    current_frame: np.ndarray,
    previous_frame: np.ndarray | None,
    current_weight: float = 0.5,
) -> np.ndarray:
    """Motion-align adjacent frames with half-resolution flow for lower cost.

    Flow is estimated on a smaller grayscale pair, then vector magnitudes are
    scaled back to full-frame pixels before the previous frame is remapped.
    This retains full-resolution output while lowering the optical-flow workload.
    """
    if current_frame is None:
        return current_frame
    if previous_frame is None or current_frame.shape != previous_frame.shape:
        return current_frame.copy()

    current_gray = _to_gray(current_frame)
    previous_gray = _to_gray(previous_frame)
    height, width = current_gray.shape[:2]
    # COLLAB_FAST_FLOW_TEMPORAL: estimate optical flow on half-resolution frames.
    flow_width = max(32, int(round(width * 0.5)))
    flow_height = max(32, int(round(height * 0.5)))
    flow_width = min(width, flow_width)
    flow_height = min(height, flow_height)
    current_small = cv2.resize(current_gray, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
    previous_small = cv2.resize(previous_gray, (flow_width, flow_height), interpolation=cv2.INTER_AREA)

    difference = float(np.mean(cv2.absdiff(current_small, previous_small)))
    if difference > 45.0:
        return current_frame.copy()

    flow_small = cv2.calcOpticalFlowFarneback(
        current_small, previous_small, None,
        0.5, 3, 15, 3, 5, 1.2, 0,
    )
    flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= width / float(flow_width)
    flow[:, :, 1] *= height / float(flow_height)

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
        current_frame, current_weight, aligned_previous, 1.0 - current_weight, 0
    )

'''


OUTPUT_PANEL = r'''    <section class="panel"><h2>06 / Motion estimation &amp; compensation output</h2>
      <p class="section-note">Four-panel result: current frame, motion-compensated prediction, amplified absolute prediction residual, and displacement vectors. Download the CSV files to inspect vectors and error measurements.</p>
      <div class="video-card"><div class="video-head"><span>Motion analysis</span><small id="motionVideoMeta">Not analysed</small></div>
        <video id="motionVideo" controls playsinline preload="metadata" class="hide"></video>
        <div id="motionEmpty" class="video-empty">Load a source video, then run motion analysis from the controls.</div>
      </div>
      <div id="motionSummary" class="message">Motion-estimation results will appear here.</div>
      <div class="transport"><a id="motionDownloadVideo" class="btn hide" href="#">Download motion-analysis MP4</a>
        <a id="motionDownloadVectors" class="btn hide" href="#">Download displacement vectors CSV</a>
        <a id="motionDownloadMetrics" class="btn hide" href="#">Download prediction-error CSV</a></div>
    </section>
'''

CONTROL_PANEL = r'''    <section class="panel"><h2>07 / Motion estimation &amp; compensation</h2>
      <p class="section-note">Runs only when you request it, so this feature does not slow ordinary filtering. Longer and higher-resolution videos take longer to analyse.</p>
      <form id="motionForm">
        <div class="controls">
          <div class="field wide"><label for="motion_algorithm">Motion-estimation technique</label>
            <select id="motion_algorithm" name="algorithm">
              <option value="block" selected>Local block matching — per-block vectors</option>
              <option value="global">Global translation — fastest for camera shift</option>
            </select>
          </div>
          <div class="field wide"><label for="motion_preset">Speed / accuracy preset</label>
            <select id="motion_preset" name="preset">
              <option value="fast" selected>Fast — recommended to start</option>
              <option value="balanced">Balanced — wider local search</option>
              <option value="accurate">Accurate — higher-resolution estimation, slower</option>
            </select>
          </div>
        </div>
        <button id="motionButton" class="btn primary" style="width:100%;margin-top:14px" type="submit" disabled>Run motion estimation</button>
      </form>
      <div id="motionMessage" class="message" role="status">Load a source video before motion analysis.</div>
      <p class="hint">Block mode records a displacement vector (dx, dy) per block using a squared-difference matching cost. Global mode estimates one translation per frame pair. A high residual means the prediction did not explain the frame well. This is a motion-analysis/prediction feature, not a replacement video codec.</p>
    </section>
'''

MOTION_JS = r'''  // Motion estimation is an opt-in analysis path, separate from ordinary filtering.
  motionForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (!jobId || busy) return;
    busy = true;
    enableAfterUpload(Boolean(jobId));
    $('motionDownloadVideo').classList.add('hide');
    $('motionDownloadVectors').classList.add('hide');
    $('motionDownloadMetrics').classList.add('hide');
    $('motionSummary').textContent = 'Estimating motion and preparing the prediction/residual video. Keep this tab open until it finishes.';
    message('motionMessage', 'Starting motion analysis…');
    status('Motion analysis running');
    try {
      const response = await fetch(`/video/motion/${encodeURIComponent(jobId)}`, {
        method: 'POST', body: new FormData(motionForm)
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || 'Motion analysis failed.');

      motionVideo.src = result.video_url + (result.video_url.includes('?') ? '&' : '?') + 'v=' + Date.now();
      motionVideo.load();
      setVideoVisible(motionVideo, 'motionEmpty', true);
      $('motionDownloadVideo').href = result.download_url;
      $('motionDownloadVideo').classList.remove('hide');
      $('motionDownloadVectors').href = result.vectors_url;
      $('motionDownloadVectors').classList.remove('hide');
      $('motionDownloadMetrics').href = result.metrics_url;
      $('motionDownloadMetrics').classList.remove('hide');

      const summary = result.summary || {};
      $('motionVideoMeta').textContent = `${summary.frames_processed ?? '—'} frames · ${summary.algorithm_key || '—'} · ${summary.elapsed_seconds ?? '—'} s`;
      $('motionSummary').textContent =
        `Method: ${summary.algorithm || '—'} | Preset: ${summary.preset || '—'}\n` +
        `Frames: ${summary.frames_processed ?? '—'} | Vector records: ${summary.vector_rows ?? '—'} | Runtime: ${summary.elapsed_seconds ?? '—'} s\n` +
        `Baseline previous-frame MAE: ${summary.mean_naive_previous_frame_mae ?? '—'}\n` +
        `Compensated-prediction MAE: ${summary.mean_compensated_prediction_mae ?? '—'}\n` +
        `Mean error improvement: ${summary.mean_error_improvement_percent ?? '—'}%\n` +
        `Positive improvement means lower average prediction error; negative means this setting performed worse than the previous-frame baseline on this video.`;
      message('motionMessage', 'Motion analysis completed. The MP4 and both CSV reports are ready.', 'good');
      status('Motion analysis complete');
    } catch (error) {
      message('motionMessage', error.message, 'error');
      $('motionSummary').textContent = 'No completed motion-analysis output is available.';
      status('Motion analysis failed', 'error');
    } finally {
      busy = false;
      enableAfterUpload(Boolean(jobId));
    }
  });

'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(APP.is_file(), f"Could not find {APP}. Extract this patch into the COLLAB project root (the folder containing app.py).")
    require(TEMPLATE.is_file(), f"Could not find {TEMPLATE}. Check that the project's templates folder is present.")
    require(PROCESSOR.is_file(), f"Could not find {PROCESSOR}. Check that app/core/video_processor.py is present.")
    require(MODULE.is_file(), f"Could not find {MODULE}. Extract all files from the patch ZIP, preserving folders.")

    app_text = APP.read_text(encoding="utf-8")
    html_text = TEMPLATE.read_text(encoding="utf-8")
    processor_text = PROCESSOR.read_text(encoding="utf-8")
    require('@app.route("/video/motion/<job_id>"' not in app_text,
            "Motion routes already exist in app.py. Nothing was changed; do not apply the patch twice.")
    require("COLLAB_FAST_FLOW_TEMPORAL" not in processor_text,
            "The temporal-flow performance patch is already installed. Nothing was changed; do not apply twice.")
    require(re.search(r"def temporal_correlation_cancel\([\s\S]*?\n(?=def process_video\()", processor_text) is not None,
            "Could not locate temporal_correlation_cancel before process_video in video_processor.py; no files were changed.")
    require('id="motionForm"' not in html_text,
            "Motion controls already exist in video.html. Nothing was changed; do not apply the patch twice.")
    require("# APPLICATION START" in app_text, "Could not find the application-start marker in app.py; no files were changed.")
    require("<aside class=\"side-column\">" in html_text, "Could not find the Video Lab layout marker in video.html; no files were changed.")
    require('id="05 / OCR text-extraction output"' not in html_text, "Unexpected video template format.")
    require('<h2>05 / OCR text-extraction output</h2>' in html_text,
            "Could not find the expected OCR output panel in video.html; no files were changed.")
    require('id="ocrForm"' in html_text and "// Native canvas bar charts" in html_text,
            "Could not find the expected OCR form or JavaScript insertion marker; no files were changed.")
    require("const uploadForm = $('uploadForm'), processForm = $('processForm'), detectForm = $('detectForm'), ocrForm = $('ocrForm');" in html_text,
            "Video Lab JavaScript has changed; expected form declaration was not found.")
    require("const original = $('originalVideo'), processed = $('processedVideo'), detection = $('detectionVideo'), ocrVideo = $('ocrVideo');" in html_text,
            "Video Lab JavaScript has changed; expected video declaration was not found.")
    require("function enableAfterUpload(enabled)" in html_text,
            "Video Lab enableAfterUpload function not found.")
    require('\n  </div>\n  <aside class="side-column">' in html_text,
            "Could not find the end of the main video-output column; no files were changed.")
    require(re.search(r'<section class="panel"><h2>\d+ / Session notes</h2>', html_text) is not None,
            "Could not find the Session notes panel; no files were changed.")

    # Prepare all modifications in memory first, so a failed anchor check cannot
    # leave one file partially patched.
    app_text = app_text.replace(
        "# APPLICATION START", ROUTES + "# APPLICATION START", 1
    )
    processor_text = re.sub(
        r"def temporal_correlation_cancel\([\s\S]*?\n(?=def process_video\()",
        FAST_TEMPORAL_FUNCTION, processor_text, count=1,
    )

    layout_anchor = '\n  </div>\n  <aside class="side-column">'
    html_text = html_text.replace(layout_anchor, '\n' + OUTPUT_PANEL + '  </div>\n  <aside class="side-column">', 1)

    session_pattern = r'(?P<indent>    )<section class="panel"><h2>\d+ / Session notes</h2>'
    require(re.search(session_pattern, html_text) is not None, "Session notes panel anchor changed; no files were changed.")
    html_text = re.sub(session_pattern, lambda m: CONTROL_PANEL + m.group("indent") + '<section class="panel"><h2>08 / Session notes</h2>', html_text, count=1)

    html_text = html_text.replace(
        "Frame processing, synchronized comparison, histograms, YOLO object detection, and OCR text extraction.",
        "Frame processing, synchronized comparison, histograms, YOLO detection, OCR, and motion estimation/compensation.",
        1,
    )
    html_text = html_text.replace(
        "const uploadForm = $('uploadForm'), processForm = $('processForm'), detectForm = $('detectForm'), ocrForm = $('ocrForm');",
        "const uploadForm = $('uploadForm'), processForm = $('processForm'), detectForm = $('detectForm'), ocrForm = $('ocrForm'), motionForm = $('motionForm');",
        1,
    )
    html_text = html_text.replace(
        "const original = $('originalVideo'), processed = $('processedVideo'), detection = $('detectionVideo'), ocrVideo = $('ocrVideo');",
        "const original = $('originalVideo'), processed = $('processedVideo'), detection = $('detectionVideo'), ocrVideo = $('ocrVideo'), motionVideo = $('motionVideo');",
        1,
    )
    enable_anchor = "$('ocrButton').disabled = !enabled || busy;"
    require(enable_anchor in html_text, "Could not modify the upload-enable function; no files were changed.")
    html_text = html_text.replace(enable_anchor, enable_anchor + " $('motionButton').disabled = !enabled || busy;", 1)
    html_text = html_text.replace("  // Native canvas bar charts: no third-party chart CDN is required.", MOTION_JS + "  // Native canvas bar charts: no third-party chart CDN is required.", 1)
    html_text = html_text.replace(
        "setVideoVisible(original,'originalEmpty',false); setVideoVisible(processed,'processedEmpty',false); setVideoVisible(ocrVideo,'ocrEmpty',false); setCompareMode();",
        "setVideoVisible(original,'originalEmpty',false); setVideoVisible(processed,'processedEmpty',false); setVideoVisible(ocrVideo,'ocrEmpty',false); setVideoVisible(motionVideo,'motionEmpty',false); setCompareMode();",
        1,
    )
    html_text = html_text.replace(
        "  ocrVideo.addEventListener('loadeddata', () => setVideoVisible(ocrVideo, 'ocrEmpty', true));",
        "  ocrVideo.addEventListener('loadeddata', () => setVideoVisible(ocrVideo, 'ocrEmpty', true));\n  motionVideo.addEventListener('loadeddata', () => setVideoVisible(motionVideo, 'motionEmpty', true));",
        1,
    )
    html_text = html_text.replace(
        "  ocrVideo.addEventListener('error',()=>message('ocrMessage','OCR output exists, but the browser could not decode it. Check H.264 support and retry.','error'));",
        "  ocrVideo.addEventListener('error',()=>message('ocrMessage','OCR output exists, but the browser could not decode it. Check H.264 support and retry.','error'));\n  motionVideo.addEventListener('error',()=>message('motionMessage','Motion-analysis output exists, but the browser could not decode it. Check H.264 support and retry.','error'));",
        1,
    )
    require("motionForm.addEventListener('submit'" in html_text and "motionButton" in html_text,
            "Internal validation failed while preparing the motion UI; no files were changed.")
    require(html_text.count('id="motionVideo"') == 1 and html_text.count('id="motionForm"') == 1,
            "Internal validation found missing or duplicate motion UI elements; no files were changed.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / f".collab_motion_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(APP, backup_dir / "app.py")
    (backup_dir / "templates").mkdir()
    (backup_dir / "app" / "core").mkdir(parents=True)
    shutil.copy2(TEMPLATE, backup_dir / "templates" / "video.html")
    shutil.copy2(PROCESSOR, backup_dir / "app" / "core" / "video_processor.py")
    try:
        APP.write_text(app_text, encoding="utf-8")
        TEMPLATE.write_text(html_text, encoding="utf-8")
        PROCESSOR.write_text(processor_text, encoding="utf-8")
    except Exception:
        shutil.copy2(backup_dir / "app.py", APP)
        shutil.copy2(backup_dir / "templates" / "video.html", TEMPLATE)
        shutil.copy2(backup_dir / "app" / "core" / "video_processor.py", PROCESSOR)
        raise

    print("COLLAB motion-estimation patch applied.")
    print(f"Backup saved in: {backup_dir}")
    print("New module: app/core/video_motion.py")
    print("Optimized temporal optical flow to estimate on half-resolution frames.")
    print("Next: run the syntax checks in README_MOTION_COMPENSATION_SETUP.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PATCH NOT APPLIED: {exc}", file=sys.stderr)
        raise SystemExit(1)
