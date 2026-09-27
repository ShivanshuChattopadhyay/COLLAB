import time

from app.core.metrics import calculate_metrics


def run_comparison(original, algorithms, processing_function, parameters=None):
    parameters = parameters or {}
    images = {}
    metrics = {}
    for algorithm in algorithms:
        start = time.perf_counter()
        result = processing_function(original.copy(), algorithm, parameters)
        elapsed = (time.perf_counter() - start) * 1000.0
        images[algorithm] = result
        data = calculate_metrics(original, result)
        data["processing_time_ms"] = elapsed
        metrics[algorithm] = data
    return {"images": images, "metrics": metrics}


def metric_leaders(metrics):
    leaders = {}
    lower_is_better = {"noise_processed", "mse", "processing_time_ms"}
    for metric in ["brightness_processed", "contrast_processed", "sharpness_processed", "noise_processed", "mse", "psnr", "ssim", "processing_time_ms"]:
        values = {name:data.get(metric) for name,data in metrics.items() if data.get(metric) is not None}
        if values:
            key = min(values, key=values.get) if metric in lower_is_better else max(values, key=values.get)
            leaders[metric] = {"algorithm": key, "value": values[key]}
    return leaders


def build_graph_data(metrics):
    names=list(metrics.keys())
    return {
        "algorithms": names,
        "brightness":[metrics[n].get("brightness_processed",0) for n in names],
        "contrast":[metrics[n].get("contrast_processed",0) for n in names],
        "sharpness":[metrics[n].get("sharpness_processed",0) for n in names],
        "noise":[metrics[n].get("noise_processed",0) for n in names],
        "mse":[metrics[n].get("mse",0) for n in names],
        "psnr":[metrics[n].get("psnr",0) for n in names],
        "ssim":[metrics[n].get("ssim",0) for n in names],
        "processing_time":[metrics[n].get("processing_time_ms",0) for n in names],
    }
