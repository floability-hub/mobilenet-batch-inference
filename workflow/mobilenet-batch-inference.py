"""Compare ordinary TaskVine PythonTask inference with a stateful library."""

# This file uses ``# %%`` markers so each section can become a notebook cell
# without changing the order or meaning of the workflow.

# %% Cell 1: imports and tutorial configuration
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import ndcctools.taskvine as vine

MODEL_PATH = Path("data/mobilenetv2-10.onnx")
LABELS_PATH = Path("data/imagenet-synset.txt")
IMAGE_DIR = Path("data/images")
MANIFEST_PATH = Path("data/image-manifest.json")
OUTPUT_DIR = Path("outputs")

BATCH_SIZE = 4
TOP_K = 5
LIBRARY_NAME = "mobilenetv2-inference"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# %% Cell 2: functions sent to ordinary PythonTask workers
def classify_batch_cold(model_path, labels_path, batch_dir, top_k):
    """Load MobileNet in this task, then classify one image microbatch."""
    import os
    import socket
    import time
    import uuid
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    started_at = time.perf_counter()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    with open(labels_path, encoding="utf-8") as labels_file:
        labels = [
            line.strip().split(" ", 1)[1]
            for line in labels_file
            if line.strip()
        ]
    loaded_at = time.perf_counter()

    predictions = []
    for image_path in sorted(Path(batch_dir).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            scale = 256.0 / min(width, height)
            resized = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.BILINEAR,
            )
            left = (resized.width - 224) // 2
            top = (resized.height - 224) // 2
            cropped = resized.crop((left, top, left + 224, top + 224))

        image_array = np.asarray(cropped, dtype=np.float32) / 255.0
        image_array = (
            image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_tensor = np.transpose(image_array, (2, 0, 1))[None, ...]

        scores = session.run(
            None,
            {session.get_inputs()[0].name: input_tensor},
        )[0].reshape(-1)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        best_indices = np.argsort(probabilities)[-top_k:][::-1]

        predictions.append(
            {
                "image": image_path.name,
                "top_predictions": [
                    {
                        "class_index": int(index),
                        "label": labels[index],
                        "probability": float(probabilities[index]),
                    }
                    for index in best_indices
                ],
            }
        )

    return {
        "predictions": predictions,
        "session_load_id": uuid.uuid4().hex[:8],
        "session_load_seconds": loaded_at - started_at,
        "task_seconds": time.perf_counter() - started_at,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


# %% Cell 3: functions used by the persistent serverless library
def load_mobilenet_library(model_path, labels_path):
    """Create one ONNX Runtime session when a library instance starts."""
    import os
    import socket
    import time
    import uuid

    import onnxruntime as ort

    started_at = time.perf_counter()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    with open(labels_path, encoding="utf-8") as labels_file:
        labels = [
            line.strip().split(" ", 1)[1]
            for line in labels_file
            if line.strip()
        ]

    return {
        "inference_session": session,
        "imagenet_labels": labels,
        "library_load_id": uuid.uuid4().hex[:8],
        "library_load_seconds": time.perf_counter() - started_at,
        "library_hostname": socket.gethostname(),
        "library_pid": os.getpid(),
    }


def classify_batch_stateful(batch_dir, top_k):
    """Classify a microbatch with the session already loaded in this process."""
    import os
    import time
    from pathlib import Path

    import numpy as np
    from ndcctools.taskvine.utils import load_variable_from_library
    from PIL import Image

    started_at = time.perf_counter()
    session = load_variable_from_library("inference_session")
    labels = load_variable_from_library("imagenet_labels")

    predictions = []
    for image_path in sorted(Path(batch_dir).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            scale = 256.0 / min(width, height)
            resized = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.BILINEAR,
            )
            left = (resized.width - 224) // 2
            top = (resized.height - 224) // 2
            cropped = resized.crop((left, top, left + 224, top + 224))

        image_array = np.asarray(cropped, dtype=np.float32) / 255.0
        image_array = (
            image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_tensor = np.transpose(image_array, (2, 0, 1))[None, ...]

        scores = session.run(
            None,
            {session.get_inputs()[0].name: input_tensor},
        )[0].reshape(-1)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        best_indices = np.argsort(probabilities)[-top_k:][::-1]

        predictions.append(
            {
                "image": image_path.name,
                "top_predictions": [
                    {
                        "class_index": int(index),
                        "label": labels[index],
                        "probability": float(probabilities[index]),
                    }
                    for index in best_indices
                ],
            }
        )

    return {
        "predictions": predictions,
        "library_load_id": load_variable_from_library("library_load_id"),
        "library_load_seconds": load_variable_from_library(
            "library_load_seconds"
        ),
        "library_hostname": load_variable_from_library("library_hostname"),
        "library_pid": load_variable_from_library("library_pid"),
        "function_pid": os.getpid(),
        "call_seconds": time.perf_counter() - started_at,
    }


# %% Cell 4: manager configuration supplied by Floability
def manager_ports():
    """Read the inclusive manager port range supplied by Floability."""
    port_spec = os.environ.get("VINE_MANAGER_PORTS", "9123,9150")
    ports = [int(value.strip()) for value in port_spec.split(",") if value.strip()]
    if not ports:
        raise ValueError("VINE_MANAGER_PORTS does not contain a port")
    if len(ports) == 1:
        return ports[0]
    return [min(ports), max(ports)]


manager_name = os.environ.get("VINE_MANAGER_NAME")
if not manager_name:
    raise RuntimeError(
        "VINE_MANAGER_NAME is not set; execute this workflow through Floability"
    )

for required_path in (MODEL_PATH, LABELS_PATH, IMAGE_DIR, MANIFEST_PATH):
    if not required_path.exists():
        raise FileNotFoundError(f"Required staged input not found: {required_path}")

manager = vine.Manager(port=manager_ports(), name=manager_name)
manager.tune("watch-library-logfiles", 1)
print(f"[manager] Name: {manager_name}")
print(f"[manager] Listening on port: {manager.port}")


# %% Cell 5: make six deterministic four-image microbatches
image_paths = sorted(
    path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
)
if len(image_paths) != 24:
    raise RuntimeError(f"Expected 24 staged images; found {len(image_paths)}")

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
manifest_by_file = {item["file"]: item for item in manifest["images"]}
if set(manifest_by_file) != {path.name for path in image_paths}:
    raise RuntimeError("The staged image set does not match image-manifest.json")
for image_path in image_paths:
    with image_path.open("rb") as image_file:
        actual_sha256 = hashlib.file_digest(image_file, "sha256").hexdigest()
    expected_sha256 = manifest_by_file[image_path.name]["local_sha256"]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {image_path.name}: expected "
            f"{expected_sha256}, received {actual_sha256}"
        )
print("[data] Verified all 24 images against image-manifest.json")

temporary_batch_root = Path(tempfile.mkdtemp(prefix="mobilenet-batches-", dir="."))
batch_paths = []
for batch_number, start in enumerate(range(0, len(image_paths), BATCH_SIZE)):
    batch_path = temporary_batch_root / f"batch-{batch_number:02d}"
    batch_path.mkdir()
    for image_path in image_paths[start : start + BATCH_SIZE]:
        shutil.copy2(image_path, batch_path / image_path.name)
    batch_paths.append(batch_path)

declared_model = manager.declare_file(str(MODEL_PATH), cache=True)
declared_labels = manager.declare_file(str(LABELS_PATH), cache=True)
declared_batches = {
    batch_path: manager.declare_file(str(batch_path), cache=True)
    for batch_path in batch_paths
}
print(
    f"[data] Prepared {len(image_paths)} images as "
    f"{len(batch_paths)} microbatches of {BATCH_SIZE}"
)


# %% Cell 6: ordinary PythonTask baseline (one model load per task)
cold_task_batches = {}
cold_started_at = time.perf_counter()

for batch_path in batch_paths:
    task = vine.PythonTask(
        classify_batch_cold,
        "model.onnx",
        "labels.txt",
        "batch",
        TOP_K,
    )
    task.add_input(declared_model, "model.onnx")
    task.add_input(declared_labels, "labels.txt")
    task.add_input(declared_batches[batch_path], "batch")
    task.set_cores(1)
    task_id = manager.submit(task)
    cold_task_batches[task_id] = batch_path.name

print(f"[cold] Submitted {len(cold_task_batches)} PythonTask microbatches")

cold_results = []
cold_failures = []
while not manager.empty():
    completed = manager.wait(5)
    if not completed:
        continue
    if not completed.successful():
        cold_failures.append((completed.id, completed.result))
        print(f"[cold] FAILED task={completed.id} result={completed.result}")
        continue

    result = completed.output
    result["task_id"] = completed.id
    result["batch"] = cold_task_batches[completed.id]
    result["worker_address"] = completed.addrport
    cold_results.append(result)
    print(
        f"[cold] task={completed.id} batch={result['batch']} "
        f"load={result['session_load_seconds']:.3f}s "
        f"worker={completed.addrport}"
    )

if cold_failures:
    raise RuntimeError(f"Cold inference task failures: {cold_failures}")
cold_elapsed = time.perf_counter() - cold_started_at


# %% Cell 7: install a stateful library that loads MobileNet once per worker
library = manager.create_library_from_functions(
    LIBRARY_NAME,
    classify_batch_stateful,
    add_env=False,
    exec_mode="direct",
    library_context_info=[
        load_mobilenet_library,
        ["model.onnx", "labels.txt"],
        {},
    ],
)
library.add_input(declared_model, "model.onnx")
library.add_input(declared_labels, "labels.txt")
library.set_cores(1)
library.set_function_slots(1)
manager.install_library(library)
print(f"[stateful] Installed persistent library: {LIBRARY_NAME}")


# %% Cell 8: submit the same six microbatches as FunctionCall tasks
stateful_task_batches = {}
stateful_started_at = time.perf_counter()

for batch_path in batch_paths:
    task = vine.FunctionCall(
        LIBRARY_NAME,
        "classify_batch_stateful",
        "batch",
        TOP_K,
    )
    task.add_input(declared_batches[batch_path], "batch")
    task.set_cores(1)
    task_id = manager.submit(task)
    stateful_task_batches[task_id] = batch_path.name

print(f"[stateful] Submitted {len(stateful_task_batches)} function calls")

stateful_results = []
stateful_failures = []
while not manager.empty():
    completed = manager.wait(5)
    if not completed:
        continue
    if not completed.successful():
        stateful_failures.append((completed.id, completed.result))
        print(f"[stateful] FAILED task={completed.id} result={completed.result}")
        continue

    result = completed.output
    if result["function_pid"] != result["library_pid"]:
        raise RuntimeError(
            "Direct function execution did not reuse the library process: "
            f"library PID {result['library_pid']}, function PID "
            f"{result['function_pid']}"
        )
    result["task_id"] = completed.id
    result["batch"] = stateful_task_batches[completed.id]
    result["worker_address"] = completed.addrport
    stateful_results.append(result)
    print(
        f"[stateful] task={completed.id} batch={result['batch']} "
        f"load_id={result['library_load_id']} "
        f"call={result['call_seconds']:.3f}s worker={completed.addrport}"
    )

if stateful_failures:
    raise RuntimeError(f"Stateful inference task failures: {stateful_failures}")
stateful_elapsed = time.perf_counter() - stateful_started_at


# %% Cell 9: validate that both execution paths returned the same predictions
def predictions_by_image(task_results):
    return {
        prediction["image"]: prediction
        for task_result in task_results
        for prediction in task_result["predictions"]
    }


cold_by_image = predictions_by_image(cold_results)
stateful_by_image = predictions_by_image(stateful_results)
if set(cold_by_image) != set(stateful_by_image) or len(cold_by_image) != 24:
    raise RuntimeError("The two execution paths did not classify the same 24 images")

for image_name in sorted(cold_by_image):
    cold_predictions = cold_by_image[image_name]["top_predictions"]
    stateful_predictions = stateful_by_image[image_name]["top_predictions"]
    if [item["class_index"] for item in cold_predictions] != [
        item["class_index"] for item in stateful_predictions
    ]:
        raise RuntimeError(f"Prediction labels differ for {image_name}")
    for cold_prediction, stateful_prediction in zip(
        cold_predictions,
        stateful_predictions,
    ):
        if abs(
            cold_prediction["probability"]
            - stateful_prediction["probability"]
        ) > 1e-6:
            raise RuntimeError(f"Prediction probabilities differ for {image_name}")

library_reuse = defaultdict(list)
for result in stateful_results:
    library_reuse[result["library_load_id"]].append(result["batch"])
if len(library_reuse) >= len(stateful_results):
    raise RuntimeError("Every FunctionCall started a new library; no state was reused")


# %% Cell 10: save machine-readable results and a labeled contact sheet
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cold_mean_task_seconds = sum(
    result["task_seconds"] for result in cold_results
) / len(cold_results)
cold_mean_load_seconds = sum(
    result["session_load_seconds"] for result in cold_results
) / len(cold_results)
stateful_mean_call_seconds = sum(
    result["call_seconds"] for result in stateful_results
) / len(stateful_results)
summary = {
    "image_count": len(image_paths),
    "batch_size": BATCH_SIZE,
    "batch_count": len(batch_paths),
    "cold_python_task": {
        "elapsed_seconds": cold_elapsed,
        "mean_task_seconds": cold_mean_task_seconds,
        "mean_session_load_seconds": cold_mean_load_seconds,
        "task_results": cold_results,
    },
    "stateful_function_call": {
        "elapsed_seconds": stateful_elapsed,
        "mean_call_seconds": stateful_mean_call_seconds,
        "distinct_library_loads": len(library_reuse),
        "task_results": stateful_results,
    },
}
summary_path = OUTPUT_DIR / "inference-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

tile_width, tile_height = 240, 220
caption_height = 52
contact_sheet = Image.new("RGB", (tile_width * 4, tile_height * 6), "white")
draw = ImageDraw.Draw(contact_sheet)
font = ImageFont.load_default()

for index, image_path in enumerate(image_paths):
    with Image.open(image_path) as source_image:
        thumbnail = source_image.convert("RGB")
        thumbnail.thumbnail((tile_width - 12, tile_height - caption_height - 12))
    x = (index % 4) * tile_width
    y = (index // 4) * tile_height
    image_x = x + (tile_width - thumbnail.width) // 2
    image_y = y + 6
    contact_sheet.paste(thumbnail, (image_x, image_y))
    prediction = stateful_by_image[image_path.name]["top_predictions"][0]
    caption = (
        f"{image_path.stem[:24]}\n"
        f"{prediction['label'][:34]}\n"
        f"{prediction['probability']:.1%}"
    )
    draw.multiline_text(
        (x + 6, y + tile_height - caption_height),
        caption,
        fill="black",
        font=font,
        spacing=2,
    )

contact_sheet_path = OUTPUT_DIR / "prediction-contact-sheet.jpg"
contact_sheet.save(contact_sheet_path, quality=90)

print("=" * 72)
print("MOBILENET BATCH INFERENCE COMPLETE")
print(f"Validated images: {len(cold_by_image)}")
print(f"PythonTask cold-load elapsed: {cold_elapsed:.2f} seconds")
print(f"Mean PythonTask time: {cold_mean_task_seconds:.3f} seconds")
print(f"Mean per-task model load: {cold_mean_load_seconds:.3f} seconds")
print(f"Stateful FunctionCall elapsed: {stateful_elapsed:.2f} seconds")
print(f"Mean FunctionCall time: {stateful_mean_call_seconds:.3f} seconds")
print(f"Distinct persistent library loads: {len(library_reuse)}")
for load_id, batches in sorted(library_reuse.items()):
    print(f"  load_id={load_id}: reused by {len(batches)} batch(es)")
print(f"Results: {summary_path}")
print(f"Contact sheet: {contact_sheet_path}")
print("=" * 72)
