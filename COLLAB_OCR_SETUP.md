# COLLAB Video Lab — Add local OCR

This add-on leaves the existing image-processing and YOLO code in place. It adds OCR as a separate operation with a preview MP4, timestamped TXT, and CSV containing text, confidence, timestamps, and bounding boxes.

## 1. Copy the new processing module

Copy `video_ocr.py` into the local project at:

`C:\Users\hp\Desktop\visionlab-ai\app\core\video_ocr.py`

Do not replace `app.py`, `video_processor.py`, or `video_detection.py`.

## 2. Install the Python wrapper

Open VS Code in `C:\Users\hp\Desktop\visionlab-ai`, then open Terminal and run:

```powershell
cd C:\Users\hp\Desktop\visionlab-ai
.\.venv\Scripts\Activate.ps1
python -m pip install pytesseract imageio-ffmpeg
```

`pytesseract` is only the Python wrapper. OCR also requires the Tesseract executable.

## 3. Install Tesseract OCR for Windows

Follow the Windows section of the Tesseract installation guide:

https://github.com/tesseract-ocr/tessdoc/blob/main/Installation.md

The guide points Windows users to the UB Mannheim Windows installers. During setup, include English language data. Hindi and English+Hindi options in the UI require Hindi language data too.

After installation, close and reopen VS Code. Run:

```powershell
tesseract --version
tesseract --list-langs
```

You should see a Tesseract version and `eng` in the language list. If the command is not recognized, add the Tesseract installation directory (usually `C:\Program Files\Tesseract-OCR`) to the Windows PATH, or define `TESSERACT_CMD` to the complete path of `tesseract.exe`. The new module checks common install paths automatically.

## 4. Add the Flask route block

Open the local `app.py`. Do not replace it with a downloaded full-file version.

Open `ocr_app_routes.txt`, copy the entire block, and paste it **after the current YOLO routes** but **before** the `# APPLICATION START` section at the bottom of `app.py`.

The code uses your existing helpers: `safe_job_id`, `get_video_job_folder`, `reserve_video_job`, `release_video_job`, and Flask imports already used by the current Video Lab. If your local `app.py` differs from this version and one of those helpers is missing, stop and share the error instead of deleting working code.

## 5. Add the Video Lab UI

Open `templates\video.html` and apply `video_ocr_html_snippets.txt` in order, sections A through G. It specifies the insertion points and the exact JavaScript additions. Do not remove or replace the current YOLO controls or handlers.

## 6. Verify and start

With the virtual environment activated, run:

```powershell
python -m py_compile app.py app\core\video_ocr.py
python app.py
```

Open `http://127.0.0.1:5000/video`, upload a short video containing clear English text, choose English and “Every 5 frames”, then select **Extract text with OCR**.

Expected outputs:
- OCR-annotated H.264 MP4 shown in the new player;
- CSV with timestamp, detected text, confidence, and bounding-box coordinates;
- TXT with timestamped recognized lines.

## Notes

- OCR is run locally; video content is not sent to a hosted OCR service.
- The route runs synchronously, so keep the tab open and start with a short video.
- “Every Nth frame” is a speed/coverage trade-off. Setting it to 1 scans every frame and is slower.
- OCR quality depends on text size, blur, contrast, camera angle, and language data. The output should be checked rather than treated as guaranteed transcription.
- The source upload is preserved. OCR writes separate output files inside the job folder.
