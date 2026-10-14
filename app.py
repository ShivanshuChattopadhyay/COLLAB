from flask import (

    Flask,

    render_template,

    request,

    redirect,

    url_for,

    send_file,

    flash,

    jsonify,

)

from pathlib import Path

from werkzeug.utils import secure_filename

import cv2

import numpy as np

import uuid

import zipfile

import shutil

import json

import os

import time

import mimetypes

import threading



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt





# ============================================================

# APP CONFIGURATION

# ============================================================



app = Flask(__name__)

app.secret_key = "collab-secret-key"



BASE_DIR = Path(__file__).resolve().parent



UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"

BATCH_FOLDER = BASE_DIR / "static" / "batch"

VIDEO_FOLDER = BASE_DIR / "static" / "video_jobs"



for folder in (

    UPLOAD_FOLDER,

    OUTPUT_FOLDER,

    BATCH_FOLDER,

    VIDEO_FOLDER,

):

    folder.mkdir(parents=True, exist_ok=True)





# ============================================================

# CORE IMPORTS

# ============================================================



from app.core.image_loader import load_image



from app.core.enhancement import (

    histogram_equalization,

    image_negative,

    log_transform,

    gamma_transform,

)



from app.core.filtering import (

    mean_filter,

    gaussian_filter,

    median_filter,

    bilateral_filter,

)



from app.core.edge_detection import (

    sobel_edge_detection,

    sobel_x,

    sobel_y,

    laplacian_edge_detection,

    canny_edge_detection,

)



from app.core.morphology import (

    erosion,

    dilation,

    opening,

    closing,

    morphological_gradient,

    top_hat,

    black_hat,

)



from app.core.frequency import (

    fft_transform,

    inverse_fft,

    low_pass_filter,

    high_pass_filter,

)



from app.core.contrast import contrast_stretch, clahe

from app.core.auto_enhancement import automatic_enhancement

from app.core.metrics import calculate_metrics

from app.core.image_enlargement import lanczos_enlarge



from app.core.batch_processor import (

    process_image as process_batch_image

)



from app.core.video_processor import (

    process_video as process_video_file

)





# Comparison functions are optional.

try:

    from app.core.comparison import (

        run_comparison as core_run_comparison,

        metric_leaders as core_metric_leaders,

        build_graph_data as core_build_graph_data,

    )

except Exception:

    core_run_comparison = None

    core_metric_leaders = None

    core_build_graph_data = None





# ============================================================

# BASIC HELPERS

# ============================================================



def get_job_folder(job_id):

    folder = OUTPUT_FOLDER / str(job_id)

    folder.mkdir(parents=True, exist_ok=True)

    return folder





def safe_job_id(job_id):

    """Validate a session ID before using it in a file path."""



    job_id = str(job_id)



    if (

        not job_id

        or Path(job_id).name != job_id

        or job_id in {".", ".."}

    ):

        raise ValueError("Invalid job ID.")



    return job_id





def save_image(image, path):

    """Save grayscale, colour, and frequency-domain outputs."""



    if image is None:

        raise ValueError(f"Cannot save an empty image: {path}")



    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)



    arr = np.asarray(image)



    if np.iscomplexobj(arr):

        arr = np.abs(arr)



    # Convert two-channel real/imaginary arrays to magnitude.

    if arr.ndim == 3 and arr.shape[2] == 2:

        real = arr[:, :, 0].astype(np.float32)

        imag = arr[:, :, 1].astype(np.float32)

        arr = np.sqrt(real * real + imag * imag)



    arr = np.nan_to_num(

        arr,

        nan=0.0,

        posinf=0.0,

        neginf=0.0,

    )



    if arr.ndim == 3 and arr.shape[2] == 1:

        arr = arr[:, :, 0]



    if arr.ndim not in (2, 3):

        raise ValueError(

            f"Unsupported image shape: {arr.shape}"

        )



    if arr.ndim == 3 and arr.shape[2] not in (3, 4):

        raise ValueError(

            f"Unsupported image channel count: {arr.shape[2]}"

        )



    if arr.dtype != np.uint8:

        arr = arr.astype(np.float32)



        if arr.size == 0:

            arr = np.zeros(arr.shape, dtype=np.uint8)

        else:

            minimum = float(arr.min())

            maximum = float(arr.max())



            if maximum > minimum:

                arr = (

                    (arr - minimum)

                    * 255.0

                    / (maximum - minimum)

                )

            else:

                arr = np.zeros_like(arr)



            arr = np.clip(arr, 0, 255).astype(np.uint8)



    if cv2.imwrite(str(path), arr):

        return path



    # Pillow fallback.

    try:

        from PIL import Image



        if arr.ndim == 2:

            Image.fromarray(arr).save(str(path))

        elif arr.shape[2] == 3:

            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            Image.fromarray(rgb).save(str(path))

        else:

            rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

            Image.fromarray(rgba).save(str(path))



        return path



    except Exception as exc:

        raise ValueError(

            f"Image writer failed for {path}: {exc}"

        ) from exc





def convert_to_gray(image):

    if image is None:

        raise ValueError("Image is empty.")



    if image.ndim == 2:

        return image



    if image.ndim == 3:

        if image.shape[2] == 1:

            return image[:, :, 0]



        if image.shape[2] == 3:

            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



        if image.shape[2] == 4:

            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)



    raise ValueError(

        f"Unsupported image shape: {image.shape}"

    )





def normalize_uint8(image):

    """Convert algorithm output into a displayable uint8 image."""



    if image is None:

        raise ValueError("Image is empty.")



    arr = np.asarray(image)



    if np.iscomplexobj(arr):

        arr = np.abs(arr)



    arr = np.nan_to_num(

        arr,

        nan=0.0,

        posinf=0.0,

        neginf=0.0,

    )



    if arr.dtype == np.uint8:

        return arr



    arr = arr.astype(np.float32)



    if arr.size == 0:

        return np.zeros_like(arr, dtype=np.uint8)



    minimum = float(arr.min())

    maximum = float(arr.max())



    if maximum > minimum:

        arr = (

            (arr - minimum)

            * 255.0

            / (maximum - minimum)

        )

    else:

        arr = np.zeros_like(arr)



    return np.clip(arr, 0, 255).astype(np.uint8)





def odd_kernel(value, default=3, minimum=1, maximum=31):

    try:

        value = int(value)

    except (TypeError, ValueError):

        value = default



    value = max(minimum, min(value, maximum))



    if value % 2 == 0:

        value += 1



    return value





def operation_filename(operation):

    return f"{operation}.png"





# ============================================================

# PIPELINE HISTORY

# ============================================================



HISTORY_JSON = "history.json"

HISTORY_TXT = "history.txt"





def read_history(job_folder):

    json_path = job_folder / HISTORY_JSON

    txt_path = job_folder / HISTORY_TXT



    if json_path.exists():

        try:

            data = json.loads(

                json_path.read_text(encoding="utf-8")

            )



            if isinstance(data, list):

                result = []



                for item in data:

                    if (

                        isinstance(item, dict)

                        and item.get("operation")

                    ):

                        result.append({

                            "operation": str(item["operation"]),

                            "parameters": item.get("parameters") or {},

                        })



                return result



        except Exception:

            pass



    result = []



    if txt_path.exists():

        for line in txt_path.read_text(

            encoding="utf-8"

        ).splitlines():

            line = line.strip()



            if line:

                result.append({

                    "operation": line,

                    "parameters": {},

                })



    return result





def write_history(job_folder, history):

    (job_folder / HISTORY_JSON).write_text(

        json.dumps(history, indent=2),

        encoding="utf-8",

    )



    (job_folder / HISTORY_TXT).write_text(

        "\n".join(

            item["operation"] for item in history

        ),

        encoding="utf-8",

    )





def history_labels(history):

    return [item["operation"] for item in history]





# ============================================================

# FILTER CALL HELPERS

# ============================================================



def call_filter(function, image, kernel_size):

    attempts = [

        lambda: function(image, kernel_size=kernel_size),

        lambda: function(image, kernel_size),

        lambda: function(image),

    ]



    last_error = None



    for attempt in attempts:

        try:

            return attempt()

        except TypeError as error:

            last_error = error



    raise last_error or RuntimeError("Filter failed.")





def call_morphology(function, image, kernel_size):

    attempts = [

        lambda: function(image, kernel_size=kernel_size),

        lambda: function(image, kernel_size),

        lambda: function(image),

    ]



    last_error = None



    for attempt in attempts:

        try:

            return attempt()

        except TypeError as error:

            last_error = error



    raise last_error or RuntimeError(

        "Morphological operation failed."

    )





# ============================================================

# IMAGE PROCESSING DISPATCHER

# ============================================================



def apply_operation(image, operation, parameters=None):

    """Shared algorithm dispatcher for images and video frames."""



    parameters = parameters or {}



    if image is None:

        raise ValueError("Image is empty.")



    operation = str(operation).strip().lower()



    aliases = {

        "mean_filter": "mean",

        "gaussian_filter": "gaussian",

        "median_filter": "median",

        "bilateral_filter": "bilateral",

        "sobel_edge_detection": "sobel",

        "laplacian_edge_detection": "laplacian",

        "canny_edge_detection": "canny",

        "histogram_equalisation": "histogram_equalization",

    }



    operation = aliases.get(operation, operation)

    gray = convert_to_gray(image)



    # Basic enhancement.

    if operation == "original":

        return image.copy()



    if operation == "grayscale":

        return gray



    if operation == "negative":

        return cv2.bitwise_not(image)



    if operation == "log":

        return log_transform(gray)



    if operation == "gamma":

        gamma = float(parameters.get("gamma", 1.0))



        if gamma <= 0:

            raise ValueError(

                "Gamma must be greater than zero."

            )



        return gamma_transform(gray, gamma)



    if operation == "histogram_equalization":

        return histogram_equalization(gray)



    if operation == "contrast_stretch":

        return contrast_stretch(gray)



    if operation == "clahe":

        return clahe(gray)



    if operation == "auto_enhancement":

        result = automatic_enhancement(image)



        if isinstance(result, dict):

            return result.get("enhanced", image)



        return result



    # Image enlargement.

    if operation in {

        "enlarge",

        "enlarge_lanczos",

        "lanczos_enlarge",

    }:

        scale = parameters.get(

            "enlarge_scale",

            parameters.get("scale", 2),

        )



        return lanczos_enlarge(image, scale)



    # Spatial filtering.

    kernel_size = odd_kernel(

        parameters.get("kernel_size", 3),

        default=3,

        minimum=1,

        maximum=31,

    )



    if operation == "mean":

        return call_filter(mean_filter, image, kernel_size)



    if operation == "gaussian":

        return call_filter(gaussian_filter, image, kernel_size)



    if operation == "median":

        return call_filter(median_filter, image, kernel_size)



    if operation == "bilateral":

        return call_filter(bilateral_filter, image, kernel_size)



    if operation == "sharpen":

        kernel = np.array(

            [

                [0, -1, 0],

                [-1, 5, -1],

                [0, -1, 0],

            ],

            dtype=np.float32,

        )



        return cv2.filter2D(image, -1, kernel)



    if operation == "unsharp":

        amount = max(

            0.0,

            float(parameters.get("amount", 1.0)),

        )



        blurred = cv2.GaussianBlur(image, (5, 5), 0)



        return cv2.addWeighted(

            image,

            1.0 + amount,

            blurred,

            -amount,

            0,

        )



    # Edge detection.

    if operation == "sobel":

        return sobel_edge_detection(gray)



    if operation == "sobel_x":

        return sobel_x(gray)



    if operation == "sobel_y":

        return sobel_y(gray)



    if operation == "laplacian":

        return laplacian_edge_detection(gray)



    if operation == "canny":

        low = int(parameters.get("lower_threshold", 50))

        high = int(parameters.get("upper_threshold", 150))



        low = max(0, min(255, low))

        high = max(0, min(255, high))



        if low >= high:

            raise ValueError(

                "Lower Canny threshold must be smaller "

                "than upper threshold."

            )



        return canny_edge_detection(gray, low, high)



    if operation == "scharr_x":

        result = cv2.Scharr(gray, cv2.CV_64F, 1, 0)

        return np.clip(np.abs(result), 0, 255).astype(np.uint8)



    if operation == "scharr_y":

        result = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        return np.clip(np.abs(result), 0, 255).astype(np.uint8)



    # Thresholding.

    if operation == "threshold":

        threshold = int(parameters.get("threshold", 127))

        threshold = max(0, min(255, threshold))



        return cv2.threshold(

            gray,

            threshold,

            255,

            cv2.THRESH_BINARY,

        )[1]



    if operation == "otsu":

        return cv2.threshold(

            gray,

            0,

            255,

            cv2.THRESH_BINARY + cv2.THRESH_OTSU,

        )[1]



    if operation == "adaptive_threshold":

        block = odd_kernel(

            parameters.get("block_size", 11),

            default=11,

            minimum=3,

            maximum=101,

        )



        c = int(parameters.get("c", 2))



        return cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY,

            block,

            c,

        )



    # Morphology.

    morphology = {

        "erosion": erosion,

        "dilation": dilation,

        "opening": opening,

        "closing": closing,

        "morphological_gradient": morphological_gradient,

        "top_hat": top_hat,

        "black_hat": black_hat,

    }



    if operation in morphology:

        return call_morphology(

            morphology[operation],

            image,

            kernel_size,

        )



    # Frequency-domain processing.

    if operation == "fft":

        spectrum = np.fft.fftshift(

            np.fft.fft2(gray.astype(np.float32))

        )



        return normalize_uint8(

            np.log1p(np.abs(spectrum))

        )



    if operation == "inverse_fft":

        spectrum = np.fft.fft2(

            gray.astype(np.float32)

        )



        return normalize_uint8(

            np.real(np.fft.ifft2(spectrum))

        )



    cutoff = max(

        1.0,

        float(parameters.get("cutoff_frequency", 30)),

    )



    if operation == "low_pass":

        return normalize_uint8(

            low_pass_filter(gray, cutoff)

        )



    if operation == "high_pass":

        return normalize_uint8(

            high_pass_filter(gray, cutoff)

        )



    # Geometric processing.

    if operation == "rotate_90":

        return cv2.rotate(

            image,

            cv2.ROTATE_90_CLOCKWISE,

        )



    if operation == "rotate_180":

        return cv2.rotate(

            image,

            cv2.ROTATE_180,

        )



    if operation == "rotate_270":

        return cv2.rotate(

            image,

            cv2.ROTATE_90_COUNTERCLOCKWISE,

        )



    if operation == "flip_horizontal":

        return cv2.flip(image, 1)



    if operation == "flip_vertical":

        return cv2.flip(image, 0)



    if operation == "resize":

        width = parameters.get("width")

        height = parameters.get("height")



        width = int(width) if width else None

        height = int(height) if height else None



        if width is None and height is None:

            return image.copy()



        old_height, old_width = image.shape[:2]



        if width is not None and height is None:

            height = max(

                1,

                round(old_height * width / old_width),

            )



        elif height is not None and width is None:

            width = max(

                1,

                round(old_width * height / old_height),

            )



        if width <= 0 or height <= 0:

            raise ValueError(

                "Resize dimensions must be positive."

            )



        return cv2.resize(

            image,

            (width, height),

            interpolation=cv2.INTER_AREA,

        )



    # Colour-channel extraction.

    if operation in {

        "red_channel",

        "green_channel",

        "blue_channel",

    }:

        if image.ndim != 3 or image.shape[2] < 3:

            raise ValueError(

                "Colour channel extraction requires a colour image."

            )



        channel = {

            "blue_channel": 0,

            "green_channel": 1,

            "red_channel": 2,

        }[operation]



        return image[:, :, channel]



    raise ValueError(

        f"Unsupported operation: {operation}"

    )





# ============================================================

# HOMEPAGE AND VIDEO LAB BUTTON

# ============================================================



@app.route("/")

def index():

    return render_template("index.html")





@app.after_request

def add_video_lab_button(response):

    """

    Add the Video Lab button to the existing homepage HTML.



    This avoids replacing or redesigning the user's index.html.

    """



    try:

        if (

            request.endpoint == "index"

            and response.content_type

            and "text/html" in response.content_type

        ):

            html = response.get_data(as_text=True)



            if (

                "video-lab-launcher" not in html

                and "</body>" in html.lower()

            ):

                css = """

                <style>

                .video-lab-launcher {

                    position: fixed;

                    top: 16px;

                    right: 22px;

                    z-index: 9999;

                    padding: 10px 15px;

                    background: #eeeeee;

                    color: #111111;

                    border: 1px solid #666666;

                    text-decoration: none;

                    font: 600 12px Segoe UI, Arial, sans-serif;

                    box-shadow: 0 3px 12px rgba(0,0,0,.3);

                }

                .video-lab-launcher:hover {

                    background: #cfcfcf;

                    color: #000000;

                }

                </style>

                """



                link = """

                <a class="video-lab-launcher"

                   href="/video">

                   Video Lab &rarr;

                </a>

                """



                if "</head>" in html.lower():

                    index = html.lower().rfind("</head>")

                    html = html[:index] + css + html[index:]



                body_index = html.lower().rfind("</body>")



                if body_index >= 0:

                    html = (

                        html[:body_index]

                        + link

                        + html[body_index:]

                    )



                response.set_data(html)



    except Exception:

        app.logger.exception(

            "Could not inject the Video Lab homepage button."

        )



    return response





# ============================================================

# UPLOAD IMAGE

# ============================================================



@app.route("/upload", methods=["POST"])

def upload():

    file = request.files.get("image")



    if file is None or not file.filename:

        flash("Please select an image.")

        return redirect(url_for("index"))



    job_id = uuid.uuid4().hex

    job_folder = get_job_folder(job_id)



    filename = Path(file.filename).name

    input_path = job_folder / filename



    try:

        file.save(str(input_path))

        image = load_image(input_path)



        save_image(image, job_folder / "original.png")

        save_image(image, job_folder / "working.png")



        write_history(job_folder, [])



    except Exception as error:

        shutil.rmtree(job_folder, ignore_errors=True)

        flash(f"Upload error: {error}")

        return redirect(url_for("index"))



    return redirect(

        url_for("workspace", job_id=job_id)

    )





# ============================================================

# IMAGE WORKSPACE

# ============================================================



@app.route("/workspace/<job_id>")

def workspace(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    job_folder = OUTPUT_FOLDER / job_id



    if not job_folder.exists():

        flash("Processing session not found.")

        return redirect(url_for("index"))



    original_path = job_folder / "original.png"

    working_path = job_folder / "working.png"



    history = read_history(job_folder)

    metrics = {}

    analysis = {}



    try:

        original = cv2.imread(

            str(original_path),

            cv2.IMREAD_UNCHANGED,

        )



        working = cv2.imread(

            str(working_path),

            cv2.IMREAD_UNCHANGED,

        )



        if original is not None and working is not None:

            metrics = calculate_metrics(original, working)



        if working is not None:

            gray = convert_to_gray(working)



            brightness = float(np.mean(gray))

            contrast = float(np.std(gray))



            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            sharpness = float(laplacian.var())



            blurred = cv2.GaussianBlur(gray, (3, 3), 0)



            noise = float(

                np.std(

                    gray.astype(np.float32)

                    - blurred.astype(np.float32)

                )

            )



            analysis = {

                "brightness": brightness,

                "contrast": contrast,

                "sharpness": sharpness,

                "noise": noise,

                "is_low_light": brightness < 80,

                "is_low_contrast": contrast < 40,

                "is_blurry": sharpness < 100,

                "is_noisy": noise > 20,

            }



    except Exception:

        app.logger.exception(

            "Workspace image-analysis calculation failed."

        )



    return render_template(

        "workspace.html",

        job_id=job_id,

        original_url=url_for(

            "output_file",

            job_id=job_id,

            filename="original.png",

        ),

        working_url=url_for(

            "output_file",

            job_id=job_id,

            filename="working.png",

        ),

        current_step=(

            history[-1]["operation"]

            if history

            else "Original"

        ),

        history=history_labels(history),

        metrics=metrics,

        analysis=analysis,

    )





# ============================================================

# PROCESS IMAGE

# ============================================================



@app.route("/process", methods=["POST"])

def process_image():

    job_id = request.form.get("job_id")

    operation = request.form.get("operation")



    if not job_id:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    if not operation:

        flash("No operation selected.")

        return redirect(

            url_for("workspace", job_id=job_id)

        )



    job_folder = OUTPUT_FOLDER / job_id

    working_path = job_folder / "working.png"



    if not working_path.exists():

        flash("Working image not found.")

        return redirect(

            url_for("workspace", job_id=job_id)

        )



    image = cv2.imread(

        str(working_path),

        cv2.IMREAD_UNCHANGED,

    )



    if image is None:

        flash("Unable to read working image.")

        return redirect(

            url_for("workspace", job_id=job_id)

        )



    parameters = {

        "gamma": request.form.get("gamma", 1.0),

        "lower_threshold": request.form.get(

            "lower_threshold", 50

        ),

        "upper_threshold": request.form.get(

            "upper_threshold", 150

        ),

        "kernel_size": request.form.get(

            "kernel_size", 3

        ),

        "cutoff_frequency": request.form.get(

            "cutoff_frequency", 30

        ),

        "threshold": request.form.get(

            "threshold", 127

        ),

        "block_size": request.form.get(

            "block_size", 11

        ),

        "c": request.form.get("c", 2),

        "amount": request.form.get("amount", 1.0),

        "width": request.form.get("width"),

        "height": request.form.get("height"),

        "enlarge_scale": request.form.get(

            "enlarge_scale",

            request.form.get("scale", "2"),

        ),

    }



    started = time.perf_counter()



    try:

        result = apply_operation(

            image,

            operation,

            parameters,

        )



        elapsed = time.perf_counter() - started



        if result is None:

            raise ValueError(

                "Processing returned no image."

            )



        result = normalize_uint8(result)



        save_image(result, working_path)

        save_image(

            result,

            job_folder / operation_filename(operation),

        )



        history = read_history(job_folder)



        history.append({

            "operation": operation,

            "parameters": parameters,

            "time": elapsed,

        })



        write_history(job_folder, history)



    except Exception as error:

        app.logger.exception(

            "Image processing failed for operation %s",

            operation,

        )

        flash(f"Processing error: {error}")



    return redirect(

        url_for("workspace", job_id=job_id)

    )





# ============================================================

# UNDO

# ============================================================



@app.route("/undo/<job_id>", methods=["GET", "POST"])

def undo(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    job_folder = OUTPUT_FOLDER / job_id

    original_path = job_folder / "original.png"

    working_path = job_folder / "working.png"



    if not original_path.exists():

        flash("Original image not found.")

        return redirect(url_for("index"))



    history = read_history(job_folder)



    if history:

        history.pop()



    image = cv2.imread(

        str(original_path),

        cv2.IMREAD_UNCHANGED,

    )



    if image is None:

        flash("Unable to rebuild the image.")

        return redirect(

            url_for("workspace", job_id=job_id)

        )



    try:

        for item in history:

            image = apply_operation(

                image,

                item["operation"],

                item.get("parameters", {}),

            )



        save_image(image, working_path)

        write_history(job_folder, history)



    except Exception as error:

        flash(f"Undo error: {error}")



    return redirect(

        url_for("workspace", job_id=job_id)

    )





# ============================================================

# RESET PIPELINE

# ============================================================



@app.route("/reset/<job_id>", methods=["GET", "POST"])

def reset(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    job_folder = OUTPUT_FOLDER / job_id

    original_path = job_folder / "original.png"

    working_path = job_folder / "working.png"



    if original_path.exists():

        shutil.copy2(original_path, working_path)

        write_history(job_folder, [])



    return redirect(

        url_for("workspace", job_id=job_id)

    )





# ============================================================

# SERVE IMAGE OUTPUT

# ============================================================



@app.route("/output/<job_id>/<path:filename>")

def output_file(job_id, filename):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        return "Invalid job ID.", 400



    job_folder = OUTPUT_FOLDER / job_id

    file_path = job_folder / filename



    try:

        file_path.resolve().relative_to(

            job_folder.resolve()

        )

    except ValueError:

        return "Invalid file path.", 400



    if not file_path.exists() or not file_path.is_file():

        return "File not found.", 404



    return send_file(file_path)





# ============================================================

# DOWNLOAD IMAGE RESULT

# ============================================================



@app.route("/download/<job_id>")

def download(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        return "Invalid job ID.", 400



    output_path = OUTPUT_FOLDER / job_id / "working.png"



    if not output_path.exists():

        return "File not found.", 404



    return send_file(

        output_path,

        as_attachment=True,

        download_name="collab_result.png",

    )





# ============================================================

# COMPARISON HELPERS

# ============================================================



COMPARISON_DEFAULTS = [

    "negative",

    "log",

    "gamma",

    "histogram_equalization",

    "clahe",

    "contrast_stretch",

    "auto_enhancement",

    "mean",

    "gaussian",

    "median",

    "bilateral",

    "sharpen",

    "unsharp",

    "sobel",

    "laplacian",

    "canny",

]





def run_local_comparison(original, algorithms):

    images = {}

    metrics = {}



    for algorithm in algorithms:

        parameters = {}



        if algorithm == "gamma":

            parameters["gamma"] = 1.2



        if algorithm == "canny":

            parameters.update({

                "lower_threshold": 50,

                "upper_threshold": 150,

            })



        started = time.perf_counter()



        try:

            result = apply_operation(

                original,

                algorithm,

                parameters,

            )



            result = normalize_uint8(result)



            elapsed_ms = (

                time.perf_counter() - started

            ) * 1000



            images[algorithm] = result



            metric_values = calculate_metrics(

                original,

                result,

            )



            metric_values["time"] = elapsed_ms

            metrics[algorithm] = metric_values



        except Exception:

            app.logger.exception(

                "Comparison failed for %s",

                algorithm,

            )



    return {

        "images": images,

        "metrics": metrics,

    }





def local_metric_leaders(metrics):

    if not metrics:

        return {}



    leaders = {}



    for metric in (

        "brightness_processed",

        "contrast_processed",

        "sharpness_processed",

        "noise_processed",

        "mse",

        "psnr",

        "ssim",

        "time",

    ):

        values = {

            name: data.get(metric)

            for name, data in metrics.items()

            if isinstance(data.get(metric), (int, float))

        }



        if not values:

            continue



        if metric in {"noise_processed", "mse", "time"}:

            leaders[metric] = min(

                values,

                key=values.get,

            )

        else:

            leaders[metric] = max(

                values,

                key=values.get,

            )



    return leaders





def local_graph_data(metrics):

    labels = list(metrics.keys())



    def values(key):

        return [

            round(float(metrics[name].get(key, 0)), 6)

            for name in labels

        ]



    return {

        "labels": labels,

        "brightness": values("brightness_processed"),

        "contrast": values("contrast_processed"),

        "sharpness": values("sharpness_processed"),

        "noise": values("noise_processed"),

        "mse": values("mse"),

        "psnr": values("psnr"),

        "ssim": values("ssim"),

        "time": values("time"),

    }





# ============================================================

# COMPARISON PAGE

# ============================================================



@app.route("/compare/<job_id>")

def compare(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    original_path = OUTPUT_FOLDER / job_id / "original.png"



    if not original_path.exists():

        flash("Original image not found.")

        return redirect(url_for("index"))



    return render_template(

        "compare.html",

        job_id=job_id,

        comparison_complete=False,

        available_algorithms=COMPARISON_DEFAULTS,

    )





@app.route("/compare/<job_id>/run", methods=["POST"])

def run_compare(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        flash("Invalid processing session.")

        return redirect(url_for("index"))



    job_folder = OUTPUT_FOLDER / job_id



    original = cv2.imread(

        str(job_folder / "original.png"),

        cv2.IMREAD_UNCHANGED,

    )



    if original is None:

        flash("Original image not found.")

        return redirect(url_for("index"))



    algorithms = request.form.getlist("algorithms")



    if not algorithms:

        flash("Select at least one algorithm.")

        return redirect(

            url_for("compare", job_id=job_id)

        )



    aliases = {

        "mean_filter": "mean",

        "gaussian_filter": "gaussian",

        "median_filter": "median",

        "bilateral_filter": "bilateral",

    }



    algorithms = [

        aliases.get(name, name)

        for name in algorithms

    ]



    if core_run_comparison is not None:

        try:

            comparison = core_run_comparison(

                original,

                algorithms,

                apply_operation,

                {},

            )

        except Exception:

            comparison = run_local_comparison(

                original,

                algorithms,

            )

    else:

        comparison = run_local_comparison(

            original,

            algorithms,

        )



    comparison_folder = job_folder / "comparison"

    comparison_folder.mkdir(

        parents=True,

        exist_ok=True,

    )



    image_urls = {}



    for algorithm, result in comparison.get("images", {}).items():

        safe_name = (

            str(algorithm)

            .replace("/", "_")

            .replace(" ", "_")

        )



        filename = f"{safe_name}.png"



        save_image(

            result,

            comparison_folder / filename,

        )



        image_urls[algorithm] = url_for(

            "comparison_output",

            job_id=job_id,

            filename=filename,

        )



    metrics = comparison.get("metrics", {})



    if core_metric_leaders is not None:

        try:

            leaders = core_metric_leaders(metrics)

        except Exception:

            leaders = local_metric_leaders(metrics)

    else:

        leaders = local_metric_leaders(metrics)



    if core_build_graph_data is not None:

        try:

            graph_data = core_build_graph_data(metrics)

        except Exception:

            graph_data = local_graph_data(metrics)

    else:

        graph_data = local_graph_data(metrics)



    return render_template(

        "compare.html",

        job_id=job_id,

        image_urls=image_urls,

        metrics=metrics,

        leaders=leaders,

        graph_data=graph_data,

        selected_algorithms=algorithms,

        available_algorithms=COMPARISON_DEFAULTS,

        comparison_complete=True,

    )





@app.route("/comparison-output/<job_id>/<filename>")

def comparison_output(job_id, filename):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        return "Invalid job ID.", 400



    folder = OUTPUT_FOLDER / job_id / "comparison"

    path = folder / filename



    try:

        path.resolve().relative_to(folder.resolve())

    except ValueError:

        return "Invalid file path.", 400



    if not path.exists() or not path.is_file():

        return "File not found.", 404



    return send_file(path)





# ============================================================

# HISTOGRAM ANALYSIS

# ============================================================



@app.route("/analysis/histogram/<job_id>")

def histogram_analysis(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        return "Invalid job ID.", 400



    job_folder = OUTPUT_FOLDER / job_id



    image = cv2.imread(

        str(job_folder / "working.png"),

        cv2.IMREAD_UNCHANGED,

    )



    if image is None:

        return "Image not found.", 404



    gray = convert_to_gray(image)



    plt.figure(figsize=(8, 4.5))

    plt.hist(gray.ravel(), bins=256, range=(0, 256))

    plt.title("COLLAB — Intensity Histogram")

    plt.xlabel("Intensity")

    plt.ylabel("Pixel Count")

    plt.tight_layout()



    path = job_folder / "histogram.png"



    plt.savefig(

        path,

        dpi=140,

        bbox_inches="tight",

    )



    plt.close()



    return send_file(path, mimetype="image/png")





# ============================================================

# FFT ANALYSIS

# ============================================================



@app.route("/analysis/fft/<job_id>")

def fft_analysis(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        return "Invalid job ID.", 400



    job_folder = OUTPUT_FOLDER / job_id



    image = cv2.imread(

        str(job_folder / "working.png"),

        cv2.IMREAD_UNCHANGED,

    )



    if image is None:

        return "Image not found.", 404



    gray = convert_to_gray(image).astype(np.float32)



    spectrum = np.fft.fftshift(

        np.fft.fft2(gray)

    )



    magnitude = 20 * np.log(

        np.abs(spectrum) + 1

    )



    plt.figure(figsize=(8, 4.5))

    plt.imshow(magnitude, cmap="gray")

    plt.axis("off")

    plt.title("COLLAB — FFT Magnitude Spectrum")

    plt.tight_layout()



    path = job_folder / "fft_spectrum.png"



    plt.savefig(

        path,

        dpi=140,

        bbox_inches="tight",

        pad_inches=0.05,

    )



    plt.close()



    return send_file(path, mimetype="image/png")





# ============================================================

# NEW IMAGE

# ============================================================



@app.route("/new")

def new_image():

    return redirect(url_for("index"))





# ============================================================

# BATCH PAGE

# ============================================================



@app.route("/batch")

def batch():

    return render_template("batch.html")





# ============================================================

# BATCH PROCESSING

# ============================================================



@app.route("/batch/process", methods=["POST"])

def batch_process():

    files = request.files.getlist("images")



    operation = request.form.get(

        "operation",

        "original",

    )



    output_format = request.form.get(

        "output_format",

        "png",

    ).lower()



    allowed_formats = {

        "png",

        "jpg",

        "jpeg",

        "bmp",

        "tif",

        "tiff",

        "webp",

    }



    if output_format not in allowed_formats:

        output_format = "png"



    if not files or not any(file.filename for file in files):

        flash("Please select at least one image.")

        return redirect(url_for("batch"))



    job_id = uuid.uuid4().hex



    job_folder = BATCH_FOLDER / job_id

    input_folder = job_folder / "input"

    output_folder = job_folder / "output"



    input_folder.mkdir(parents=True, exist_ok=True)

    output_folder.mkdir(parents=True, exist_ok=True)



    parameters = {

        "threshold": request.form.get("threshold", 127),

        "gamma": request.form.get("gamma", 1.0),

        "kernel_size": request.form.get("kernel_size", 3),

        "lower_threshold": request.form.get(

            "lower_threshold", 50

        ),

        "upper_threshold": request.form.get(

            "upper_threshold", 150

        ),

        "cutoff_frequency": request.form.get(

            "cutoff_frequency", 30

        ),

        "block_size": request.form.get("block_size", 11),

        "c": request.form.get("c", 2),

        "width": request.form.get("width"),

        "height": request.form.get("height"),

        "enlarge_scale": request.form.get(

            "enlarge_scale",

            request.form.get("scale", "2"),

        ),

    }



    processed_count = 0

    failed_count = 0

    output_files = []

    errors = []



    for index, file in enumerate(files, start=1):

        if not file.filename:

            continue



        filename = Path(file.filename).name

        input_path = input_folder / filename



        try:

            file.save(str(input_path))



            image = cv2.imread(

                str(input_path),

                cv2.IMREAD_UNCHANGED,

            )



            if image is None:

                raise ValueError("Unable to read image.")



            try:

                result = apply_operation(

                    image,

                    operation,

                    parameters,

                )

            except Exception as primary_error:

                try:

                    result = process_batch_image(

                        image,

                        operation,

                        parameters,

                    )

                except Exception:

                    raise primary_error



            result = normalize_uint8(result)



            output_name = (

                f"{index:04d}_{Path(filename).stem}."

                f"{output_format}"

            )



            save_image(

                result,

                output_folder / output_name,

            )



            output_files.append(output_name)

            processed_count += 1



        except Exception as error:

            failed_count += 1



            errors.append({

                "filename": filename,

                "error": str(error),

            })



            app.logger.exception(

                "Batch processing failed for %s",

                filename,

            )



    summary = {

        "job_id": job_id,

        "operation": operation,

        "output_format": output_format,

        "processed_count": processed_count,

        "failed_count": failed_count,

        "output_files": output_files,

        "errors": errors,

    }



    (job_folder / "summary.json").write_text(

        json.dumps(summary, indent=2),

        encoding="utf-8",

    )



    return render_template(

        "batch_result.html",

        job_id=job_id,

        operation=operation,

        processed_count=processed_count,

        failed_count=failed_count,

        output_files=output_files,

        errors=errors,

    )





# ============================================================

# DOWNLOAD BATCH RESULTS

# ============================================================



@app.route("/batch/download/<job_id>")

def batch_download(job_id):

    try:

        job_id = safe_job_id(job_id)

    except ValueError:

        return "Invalid job ID.", 400



    job_folder = BATCH_FOLDER / job_id

    output_folder = job_folder / "output"



    if not output_folder.exists():

        return "Batch output not found.", 404



    zip_path = BATCH_FOLDER / f"{job_id}.zip"



    if zip_path.exists():

        zip_path.unlink()



    with zipfile.ZipFile(

        zip_path,

        "w",

        zipfile.ZIP_DEFLATED,

    ) as archive:

        for file_path in output_folder.iterdir():

            if file_path.is_file():

                archive.write(

                    file_path,

                    arcname=file_path.name,

                )



    return send_file(

        zip_path,

        as_attachment=True,

        download_name="collab_batch_results.zip",

    )





# ============================================================

# VIDEO LAB SETTINGS
# ============================================================

VIDEO_ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
}

VIDEO_OPERATIONS = {
    "original", "grayscale", "negative", "log", "gamma",
    "histogram_equalization", "contrast_stretch", "clahe",
    "auto_enhancement", "mean", "gaussian", "median", "bilateral",
    "sharpen", "unsharp", "sobel", "sobel_x", "sobel_y", "laplacian",
    "scharr_x", "scharr_y", "canny", "threshold", "otsu",
    "adaptive_threshold", "erosion", "dilation", "opening", "closing",
    "morphological_gradient", "top_hat", "black_hat", "fft",
    "inverse_fft", "low_pass", "high_pass",
}

VIDEO_SPEEDS = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}

# Flask may serve multiple requests concurrently. These locks protect local
# job reservations and atomic status writes in the current server process.
_VIDEO_JOB_LOCK = threading.Lock()
_VIDEO_STATUS_LOCK = threading.RLock()
_ACTIVE_VIDEO_JOBS = set()


def reserve_video_job(job_id):
    """Return False if this job is already being processed in this process."""
    with _VIDEO_JOB_LOCK:
        if job_id in _ACTIVE_VIDEO_JOBS:
            return False
        _ACTIVE_VIDEO_JOBS.add(job_id)
        return True


def release_video_job(job_id):
    with _VIDEO_JOB_LOCK:
        _ACTIVE_VIDEO_JOBS.discard(job_id)


def get_video_job_folder(job_id):
    job_id = safe_job_id(job_id)
    folder = VIDEO_FOLDER / job_id
    try:
        folder.resolve().relative_to(VIDEO_FOLDER.resolve())
    except ValueError as error:
        raise ValueError("Invalid video job path.") from error
    return folder


def write_video_status(job_folder, **updates):
    """Write status atomically; reset stale result fields when a run starts."""
    path = job_folder / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with _VIDEO_STATUS_LOCK:
        starting_new_run = (
            updates.get("state") == "processing"
            and updates.get("frames_processed") == 0
        )

        status = {}
        if not starting_new_run and path.exists():
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = {}

        status.update(updates)
        status["updated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime()
        )

        temporary_path = path.with_name("status.json.tmp")
        try:
            temporary_path.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


def read_video_status(job_folder):
    path = job_folder / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# VIDEO LAB PAGE
# ============================================================

@app.route("/video")
def video_lab():
    return render_template("video.html")


# UPLOAD SOURCE VIDEO
# ============================================================

@app.route("/video/upload", methods=["POST"])
def video_upload():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "Select a video file first."}), 400

    filename = secure_filename(uploaded_file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in VIDEO_ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": "Unsupported video file extension."}), 400

    job_id = uuid.uuid4().hex
    job_folder = VIDEO_FOLDER / job_id
    job_folder.mkdir(parents=True, exist_ok=True)
    input_path = job_folder / f"input{extension}"

    try:
        uploaded_file.save(str(input_path))
        capture = cv2.VideoCapture(str(input_path))
        try:
            if not capture.isOpened():
                raise ValueError("OpenCV cannot open this video's codec.")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

        if width <= 0 or height <= 0:
            raise ValueError("Could not read the video dimensions.")

        duration = total_frames / fps if fps > 0 and total_frames > 0 else None
        write_video_status(
            job_folder,
            state="uploaded",
            message="Video loaded and ready.",
            frames_processed=0,
            total_frames=total_frames,
            percent=0,
        )

        return jsonify({
            "ok": True,
            "job_id": job_id,
            "input_url": url_for("video_file", job_id=job_id, filename=input_path.name),
            "metadata": {
                "width": width,
                "height": height,
                "fps": round(fps, 3) if fps > 0 else None,
                "frames": total_frames,
                "duration": round(duration, 2) if duration is not None else None,
            },
        })

    except Exception as error:
        shutil.rmtree(job_folder, ignore_errors=True)
        app.logger.exception("Video upload failed")
        return jsonify({"ok": False, "error": str(error)}), 400


# PROCESS VIDEO FRAME BY FRAME
# ============================================================

@app.route("/video/process/<job_id>", methods=["POST"])
def video_process(job_id):
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid video job."}), 400

    input_files = sorted(job_folder.glob("input.*"))
    if not input_files:
        return jsonify({"ok": False, "error": "Source video was not found."}), 404

    operation = request.form.get("operation", "original").strip().lower()
    if operation not in VIDEO_OPERATIONS:
        return jsonify({"ok": False, "error": "Unsupported video operation."}), 400

    try:
        playback_speed = float(request.form.get("playback_speed", "1"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid playback speed."}), 400

    if playback_speed not in VIDEO_SPEEDS:
        return jsonify({"ok": False, "error": "Choose a supported playback speed."}), 400

    temporal_cancel = request.form.get("temporal_cancel") == "on"
    force_grayscale = request.form.get("force_grayscale") == "on"
    parameters = {
        "gamma": request.form.get("gamma", "1.0"),
        "kernel_size": request.form.get("kernel_size", "3"),
        "lower_threshold": request.form.get("lower_threshold", "50"),
        "upper_threshold": request.form.get("upper_threshold", "150"),
        "threshold": request.form.get("threshold", "127"),
        "block_size": request.form.get("block_size", "11"),
        "c": request.form.get("c", "2"),
        "cutoff_frequency": request.form.get("cutoff_frequency", "30"),
        "amount": request.form.get("amount", "1.0"),
    }

    if not reserve_video_job(job_id):
        return jsonify({
            "ok": False,
            "state": "processing",
            "error": "This video is already processing. Please wait for it to finish.",
        }), 409

    output_path = job_folder / "processed.mp4"
    try:
        # This reset removes stale video_info, output URLs, and errors from older runs.
        write_video_status(
            job_folder,
            state="processing",
            message="Starting frame-by-frame processing.",
            frames_processed=0,
            total_frames=None,
            percent=0,
            operation=operation,
            playback_speed=playback_speed,
            temporal_cancel=temporal_cancel,
            force_grayscale=force_grayscale,
        )

        def progress_callback(frames_processed, total_frames):
            percent = (
                min(100, round(frames_processed * 100 / total_frames))
                if total_frames > 0 else None
            )
            write_video_status(
                job_folder,
                state="processing",
                message=f"Processed {frames_processed} of {total_frames} frames.",
                frames_processed=frames_processed,
                total_frames=total_frames,
                percent=percent,
            )

        result = process_video_file(
            input_path=input_files[0],
            output_path=output_path,
            operation=operation,
            apply_operation_fn=apply_operation,
            parameters=parameters,
            temporal_cancel=temporal_cancel,
            force_grayscale=force_grayscale,
            playback_speed=playback_speed,
            progress_callback=progress_callback,
        )

        video_url = url_for("video_file", job_id=job_id, filename="processed.mp4")
        download_url = url_for("video_download", job_id=job_id)
        write_video_status(
            job_folder,
            state="completed",
            message="Video processing completed.",
            frames_processed=result["frames_processed"],
            total_frames=result["total_frames_reported"],
            percent=100,
            output_url=video_url,
            download_url=download_url,
            video_info=result,
        )
        return jsonify({
            "ok": True,
            "state": "completed",
            "message": "Video processing completed.",
            "video_url": video_url,
            "download_url": download_url,
            "video_info": result,
        })

    except Exception as error:
        app.logger.exception("Video processing failed for %s", job_id)
        try:
            write_video_status(
                job_folder,
                state="failed",
                message=str(error),
                error=str(error),
            )
        except Exception:
            app.logger.exception("Unable to save failed video status for %s", job_id)
        return jsonify({"ok": False, "state": "failed", "error": str(error)}), 500

    finally:
        release_video_job(job_id)


# VIDEO PROCESSING STATUS
# ============================================================

@app.route("/video/status/<job_id>")
def video_status(job_id):
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return jsonify({"error": "Invalid video job."}), 400

    status = read_video_status(job_folder)
    if status is None:
        return jsonify({"error": "Video processing status not found."}), 404

    # If Flask restarted while a job was running, no worker remains to finish it.
    if status.get("state") == "processing":
        with _VIDEO_JOB_LOCK:
            active = job_id in _ACTIVE_VIDEO_JOBS
        if not active:
            status = {
                **status,
                "state": "failed",
                "message": (
                    "Processing was interrupted because the server stopped or restarted. "
                    "Upload the video again and retry."
                ),
                "error": "Video job was interrupted.",
            }
            write_video_status(job_folder, **status)

    return jsonify(status)


# SERVE INPUT OR COMPLETED VIDEO
# ============================================================

@app.route("/video/file/<job_id>/<path:filename>")
def video_file(job_id, filename):
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400

    allowed = filename == "processed.mp4"
    if filename.startswith("input."):
        allowed = Path(filename).suffix.lower() in VIDEO_ALLOWED_EXTENSIONS
    if not allowed or Path(filename).name != filename:
        return "Invalid video filename.", 400

    file_path = job_folder / filename
    if not file_path.exists() or not file_path.is_file():
        return "Video file not found.", 404

    # Never serve a processed file while it is being written/finalized.
    if filename == "processed.mp4":
        status = read_video_status(job_folder) or {}
        if status.get("state") != "completed":
            return "Processed video is not ready yet.", 409

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return send_file(file_path, mimetype=content_type, conditional=True)


# DOWNLOAD COMPLETED VIDEO
# ============================================================

@app.route("/video/download/<job_id>")
def video_download(job_id):
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400

    status = read_video_status(job_folder) or {}
    output_path = job_folder / "processed.mp4"
    if status.get("state") != "completed" or not output_path.is_file():
        return "Processed video is not ready yet.", 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name="collab_processed_video.mp4",
        conditional=True,
    )



# ============================================================
# OPTIONAL YOLO VIDEO OBJECT DETECTION
# ============================================================

def write_detection_status(job_folder, **updates):
    """Keep YOLO run status separate from the ordinary video-processing status."""
    path = job_folder / "detection_status.json"
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


@app.route("/video/detect/<job_id>", methods=["POST"])
def video_detect(job_id):
    """Run YOLO object detection over the uploaded source video."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid video job."}), 400

    input_files = sorted(job_folder.glob("input.*"))
    if not input_files:
        return jsonify({"ok": False, "error": "Source video was not found. Upload it again."}), 404

    try:
        confidence = float(request.form.get("confidence", "0.25"))
        image_size = int(request.form.get("image_size", "640"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Confidence and image size must be numeric."}), 400

    model_name = request.form.get("model_name", "yolo26n.pt").strip()
    class_filter = request.form.get("class_filter", "").strip()
    if model_name not in {"yolo26n.pt", "yolo26s.pt"}:
        return jsonify({"ok": False, "error": "Choose yolo26n.pt or yolo26s.pt."}), 400
    if not 0.05 <= confidence <= 0.95:
        return jsonify({"ok": False, "error": "Confidence must be between 0.05 and 0.95."}), 400
    if image_size not in {320, 480, 640, 960}:
        return jsonify({"ok": False, "error": "Image size must be 320, 480, 640, or 960."}), 400
    if not reserve_video_job(job_id):
        return jsonify({"ok": False, "error": "This video is already being processed. Wait for that operation to finish."}), 409

    output_path = job_folder / "detected.mp4"
    csv_path = job_folder / "detections.csv"
    try:
        write_detection_status(
            job_folder, state="processing", message="YOLO is analysing each frame.",
            error=None, output_url=None, csv_url=None, summary=None,
        )
        from app.core.video_detection import detect_video
        result = detect_video(
            input_path=input_files[0],
            output_path=output_path,
            csv_path=csv_path,
            confidence=confidence,
            image_size=image_size,
            model_name=model_name,
            class_filter=class_filter,
        )
        video_url = url_for("video_detection_file", job_id=job_id)
        csv_url = url_for("video_detection_csv", job_id=job_id)
        download_url = url_for("video_detection_download", job_id=job_id)
        write_detection_status(
            job_folder, state="completed", message="YOLO detection completed.",
            output_url=video_url, download_url=download_url, csv_url=csv_url, summary=result,
        )
        return jsonify({
            "ok": True, "message": "YOLO detection completed.",
            "video_url": video_url, "download_url": download_url,
            "csv_url": csv_url, "summary": result,
        })
    except Exception as error:
        app.logger.exception("YOLO video detection failed for %s", job_id)
        try:
            write_detection_status(job_folder, state="failed", message="YOLO detection failed.", error=str(error))
        except Exception:
            app.logger.exception("Could not save YOLO detection status for %s", job_id)
        return jsonify({"ok": False, "error": str(error)}), 500
    finally:
        release_video_job(job_id)


@app.route("/video/detect/file/<job_id>")
def video_detection_file(job_id):
    """Stream the completed annotated YOLO MP4 to the in-page player."""
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_path = job_folder / "detection_status.json"
    output_path = job_folder / "detected.mp4"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if status.get("state") != "completed" or not output_path.is_file():
        return "YOLO detection output is not ready yet.", 409
    return send_file(output_path, mimetype="video/mp4", conditional=True)


@app.route("/video/detect/download/<job_id>")
def video_detection_download(job_id):
    """Download the annotated YOLO video."""
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_path = job_folder / "detection_status.json"
    output_path = job_folder / "detected.mp4"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if status.get("state") != "completed" or not output_path.is_file():
        return "YOLO detection output is not ready yet.", 404
    return send_file(output_path, as_attachment=True, download_name="collab_yolo_detected.mp4", mimetype="video/mp4", conditional=True)


@app.route("/video/detect/csv/<job_id>")
def video_detection_csv(job_id):
    """Download frame-level YOLO detection boxes as CSV."""
    try:
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_path = job_folder / "detection_status.json"
    csv_path = job_folder / "detections.csv"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if status.get("state") != "completed" or not csv_path.is_file():
        return "YOLO detection CSV is not ready yet.", 404
    return send_file(csv_path, as_attachment=True, download_name="collab_yolo_detections.csv", mimetype="text/csv", conditional=True)




# ============================================================
# LOCAL VIDEO OCR (TESSERACT)
# Separate from ordinary processing and YOLO detection.
# ============================================================

def write_ocr_status(job_folder, **updates):
    """Persist OCR status in its own JSON file without touching other status."""
    path = job_folder / "ocr_status.json"
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


def read_ocr_status(job_folder):
    path = job_folder / "ocr_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.route("/video/ocr/<job_id>", methods=["POST"])
def video_ocr(job_id):
    """Extract text from sampled source-video frames and export result files."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid video job."}), 400

    input_files = sorted(job_folder.glob("input.*"))
    if not input_files:
        return jsonify({
            "ok": False,
            "error": "Source video was not found. Upload it again.",
        }), 404

    try:
        frame_stride = int(request.form.get("frame_stride", "5"))
        min_confidence = float(request.form.get("min_confidence", "35"))
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "Frame interval and OCR confidence must be numeric.",
        }), 400

    language = request.form.get("language", "eng").strip()
    if language not in {"eng", "hin", "eng+hin"}:
        return jsonify({
            "ok": False,
            "error": "Choose English, Hindi, or English + Hindi.",
        }), 400
    if not 1 <= frame_stride <= 30:
        return jsonify({
            "ok": False,
            "error": "Frame interval must be from 1 to 30.",
        }), 400
    if not 0 <= min_confidence <= 100:
        return jsonify({
            "ok": False,
            "error": "OCR confidence must be from 0 to 100.",
        }), 400

    # Share the existing job lock with video processing and YOLO so operations
    # cannot overwrite one another or overload a single job at the same time.
    if not reserve_video_job(job_id):
        return jsonify({
            "ok": False,
            "error": "This video is already being processed. Wait for it to finish.",
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

        # Import lazily so COLLAB can still start if OCR dependencies are not installed.
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
            error=None,
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
    """Stream the OCR-annotated H.264 preview in the browser."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_data = read_ocr_status(job_folder)
    output_path = job_folder / "ocr_detected.mp4"
    if status_data.get("state") != "completed" or not output_path.is_file():
        return "OCR output is not ready yet.", 404
    return send_file(output_path, mimetype="video/mp4", conditional=True)


@app.route("/video/ocr/download/<job_id>")
def video_ocr_download(job_id):
    """Download the OCR-annotated H.264 preview."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_data = read_ocr_status(job_folder)
    output_path = job_folder / "ocr_detected.mp4"
    if status_data.get("state") != "completed" or not output_path.is_file():
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
    """Download recognized lines, confidence, timestamps, and boxes as CSV."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_data = read_ocr_status(job_folder)
    csv_path = job_folder / "ocr_results.csv"
    if status_data.get("state") != "completed" or not csv_path.is_file():
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
    """Download recognized text with frame timestamps."""
    try:
        job_id = safe_job_id(job_id)
        job_folder = get_video_job_folder(job_id)
    except ValueError:
        return "Invalid video job.", 400
    status_data = read_ocr_status(job_folder)
    text_path = job_folder / "ocr_results.txt"
    if status_data.get("state") != "completed" or not text_path.is_file():
        return "OCR text is not ready yet.", 404
    return send_file(
        text_path,
        as_attachment=True,
        download_name="collab_ocr_results.txt",
        mimetype="text/plain; charset=utf-8",
        conditional=True,
    )


# ============================================================
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


# APPLICATION START

# ============================================================



if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000,

    )