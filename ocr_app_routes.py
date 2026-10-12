# ============================================================
# OCR VIDEO EXTRACTION ROUTES
# Paste this whole block near the bottom of app.py, AFTER the existing
# YOLO detection routes and BEFORE the '# APPLICATION START' section.
# ============================================================

def write_ocr_status(job_folder, **updates):
    """Store OCR state separately from video-processing and YOLO state."""
    path = job_folder / "ocr_status.json"
    status = {}
    if path.exists():
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
    status.update(updates)
    status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_ocr_status(job_folder):
    path = job_folder / "ocr_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.route("/video/ocr/<job_id>", methods=["POST"])
def video_ocr(job_id):
    """Extract text from sampled frames and export an annotated video and results."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid video job."}), 400

    input_files = sorted(job_folder.glob("input.*"))
    if not input_files:
        return jsonify({"ok": False, "error": "Source video was not found. Upload it again."}), 404

    try:
        frame_stride = int(request.form.get("frame_stride", "5"))
        min_confidence = float(request.form.get("min_confidence", "35"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Frame interval and confidence must be numeric."}), 400

    language = request.form.get("language", "eng").strip()
    if not 1 <= frame_stride <= 30:
        return jsonify({"ok": False, "error": "Frame interval must be from 1 to 30."}), 400
    if not 0 <= min_confidence <= 100:
        return jsonify({"ok": False, "error": "OCR confidence must be from 0 to 100."}), 400
    if language not in {"eng", "hin", "eng+hin"}:
        return jsonify({"ok": False, "error": "Choose English, Hindi, or English + Hindi."}), 400

    if not reserve_video_job(job_id):
        return jsonify({
            "ok": False,
            "error": "This video is already being processed. Wait for that operation to finish.",
        }), 409

    output_path = job_folder / "ocr_detected.mp4"
    csv_path = job_folder / "ocr_results.csv"
    text_path = job_folder / "ocr_results.txt"

    try:
        write_ocr_status(
            job_folder,
            state="processing",
            message="OCR is scanning sampled video frames.",
            error=None,
            output_url=None,
            download_url=None,
            csv_url=None,
            text_url=None,
            summary=None,
        )
        from app.core.video_ocr import extract_video_text

        result = extract_video_text(
            input_path=input_files[0],
            output_path=output_path,
            csv_path=csv_path,
            text_path=text_path,
            language=language,
            frame_stride=frame_stride,
            min_confidence=min_confidence,
        )

        video_url = url_for("video_ocr_file", job_id=job_id)
        download_url = url_for("video_ocr_download", job_id=job_id)
        csv_url = url_for("video_ocr_csv", job_id=job_id)
        text_url = url_for("video_ocr_text", job_id=job_id)
        write_ocr_status(
            job_folder,
            state="completed",
            message="OCR extraction completed.",
            output_url=video_url,
            download_url=download_url,
            csv_url=csv_url,
            text_url=text_url,
            summary=result,
        )
        return jsonify({
            "ok": True,
            "message": "OCR extraction completed.",
            "video_url": video_url,
            "download_url": download_url,
            "csv_url": csv_url,
            "text_url": text_url,
            "summary": result,
        })
    except Exception as error:
        app.logger.exception("Video OCR failed for %s", job_id)
        try:
            write_ocr_status(
                job_folder,
                state="failed",
                message="OCR extraction failed.",
                error=str(error),
            )
        except Exception:
            app.logger.exception("Could not save OCR status for %s", job_id)
        return jsonify({"ok": False, "error": str(error)}), 500
    finally:
        release_video_job(job_id)


@app.route("/video/ocr/file/<job_id>")
def video_ocr_file(job_id):
    """Stream the OCR-annotated video in Video Lab."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status = read_ocr_status(job_folder)
    output_path = job_folder / "ocr_detected.mp4"
    if status.get("state") != "completed" or not output_path.is_file():
        return "OCR output is not ready yet.", 404
    return send_file(output_path, mimetype="video/mp4", conditional=True)


@app.route("/video/ocr/download/<job_id>")
def video_ocr_download(job_id):
    """Download the OCR-annotated video."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status = read_ocr_status(job_folder)
    output_path = job_folder / "ocr_detected.mp4"
    if status.get("state") != "completed" or not output_path.is_file():
        return "OCR output is not ready yet.", 404
    return send_file(
        output_path,
        as_attachment=True,
        download_name="collab_ocr_annotated.mp4",
        mimetype="video/mp4",
        conditional=True,
    )


@app.route("/video/ocr/csv/<job_id>")
def video_ocr_csv(job_id):
    """Download line-level OCR results, timestamps, and bounding boxes as CSV."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status = read_ocr_status(job_folder)
    csv_path = job_folder / "ocr_results.csv"
    if status.get("state") != "completed" or not csv_path.is_file():
        return "OCR CSV is not ready yet.", 404
    return send_file(
        csv_path,
        as_attachment=True,
        download_name="collab_ocr_results.csv",
        mimetype="text/csv",
        conditional=True,
    )


@app.route("/video/ocr/text/<job_id>")
def video_ocr_text(job_id):
    """Download recognized text with timestamps as a plain text file."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status = read_ocr_status(job_folder)
    text_path = job_folder / "ocr_results.txt"
    if status.get("state") != "completed" or not text_path.is_file():
        return "OCR text is not ready yet.", 404
    return send_file(
        text_path,
        as_attachment=True,
        download_name="collab_ocr_results.txt",
        mimetype="text/plain; charset=utf-8",
        conditional=True,
    )
