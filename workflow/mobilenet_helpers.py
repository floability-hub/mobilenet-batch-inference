"""Shared data and inference helpers for the MobileNet tutorial workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import ceil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_and_verify_images(
    image_dir: Path | str,
    manifest_path: Path | str,
) -> tuple[list[Path], dict]:
    """Validate a manifest and return its images in deterministic filename order.

    Manifest ``file`` values are paths relative to ``image_dir``. The current
    contract deliberately requires flat, unique filenames because TaskVine
    microbatches expose all of their images in one directory.
    """
    image_dir = Path(image_dir)
    manifest_path = Path(manifest_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Staged image directory not found: {image_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Staged image manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported image manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    entries = manifest.get("images")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Image manifest must contain a non-empty 'images' list")

    manifest_count = manifest.get("image_count")
    if manifest_count != len(entries):
        raise ValueError(
            f"Manifest image_count is {manifest_count!r}, but it contains "
            f"{len(entries)} image entries"
        )

    entries_by_file = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every image manifest entry must be an object")
        filename = entry.get("file")
        checksum = entry.get("sha256")
        if not isinstance(filename, str) or not filename:
            raise ValueError("Every image entry must have a non-empty 'file'")
        if Path(filename).name != filename:
            raise ValueError(
                f"Image manifest file must be a flat relative filename: {filename!r}"
            )
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension in manifest: {filename}")
        if filename in entries_by_file:
            raise ValueError(f"Duplicate image filename in manifest: {filename}")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"Missing or invalid SHA-256 for {filename}")
        entries_by_file[filename] = entry

    staged_files = {
        path.name: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if set(staged_files) != set(entries_by_file):
        missing = sorted(set(entries_by_file) - set(staged_files))
        unexpected = sorted(set(staged_files) - set(entries_by_file))
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            "Staged images do not match the manifest: " + ", ".join(details)
        )

    image_paths = []
    for filename in sorted(entries_by_file):
        image_path = staged_files[filename]
        with image_path.open("rb") as image_file:
            actual_sha256 = hashlib.file_digest(image_file, "sha256").hexdigest()
        expected_sha256 = entries_by_file[filename]["sha256"]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )
        image_paths.append(image_path)

    return image_paths, manifest


def make_image_batches(
    image_paths: Sequence[Path],
    batch_size: int,
) -> list[list[Path]]:
    """Divide deterministic image paths into deterministic microbatches."""
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be at least 1")
    if not image_paths:
        raise ValueError("Cannot create batches from an empty image collection")
    return [
        list(image_paths[start : start + batch_size])
        for start in range(0, len(image_paths), batch_size)
    ]


def materialize_batch_directories(
    batches: Sequence[Sequence[Path]],
    *,
    prefix: str = "mobilenet-batches-",
) -> tuple[Path, list[Path]]:
    """Copy image batches into directories that can be declared to TaskVine."""
    batch_root = Path(tempfile.mkdtemp(prefix=prefix, dir="."))
    batch_paths = []
    for batch_number, image_batch in enumerate(batches):
        batch_path = batch_root / f"batch-{batch_number:05d}"
        batch_path.mkdir()
        for image_path in image_batch:
            shutil.copy2(image_path, batch_path / image_path.name)
        batch_paths.append(batch_path)
    return batch_root, batch_paths


def create_inference_session(model_path):
    """Create a single-threaded CPU ONNX Runtime session."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def load_imagenet_labels(labels_path):
    """Load the ImageNet index-to-label mapping used by MobileNet."""
    with open(labels_path, encoding="utf-8") as labels_file:
        return [
            line.strip().split(" ", 1)[1]
            for line in labels_file
            if line.strip()
        ]


def classify_image_directory(session, labels, batch_dir, top_k):
    """Preprocess and classify every supported image in one directory."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

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

    return predictions


def classify_batch_with_new_session(model_path, labels_path, batch_dir, top_k):
    """Load a new MobileNet session and classify one image microbatch."""
    import os
    import socket
    import time
    import uuid

    started_at = time.perf_counter()
    session = create_inference_session(model_path)
    labels = load_imagenet_labels(labels_path)
    loaded_at = time.perf_counter()
    predictions = classify_image_directory(session, labels, batch_dir, top_k)

    return {
        "predictions": predictions,
        "session_load_id": uuid.uuid4().hex[:8],
        "session_load_seconds": loaded_at - started_at,
        "task_seconds": time.perf_counter() - started_at,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def initialize_mobilenet_library(model_path, labels_path):
    """Initialize the reusable state for one MobileNet library process."""
    import os
    import socket
    import time
    import uuid

    started_at = time.perf_counter()
    session = create_inference_session(model_path)
    labels = load_imagenet_labels(labels_path)

    return {
        "inference_session": session,
        "imagenet_labels": labels,
        "library_load_id": uuid.uuid4().hex[:8],
        "library_load_seconds": time.perf_counter() - started_at,
        "library_hostname": socket.gethostname(),
        "library_pid": os.getpid(),
    }


def classify_batch_with_shared_session(batch_dir, top_k):
    """Classify a microbatch with the library's shared MobileNet session."""
    import os
    import time

    from ndcctools.taskvine.utils import load_variable_from_library

    started_at = time.perf_counter()
    session = load_variable_from_library("inference_session")
    labels = load_variable_from_library("imagenet_labels")
    predictions = classify_image_directory(session, labels, batch_dir, top_k)

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


def predictions_by_image(
    task_results: Iterable[dict],
    expected_image_names: Iterable[str],
) -> dict[str, dict]:
    """Flatten task results and require exactly one result per input image."""
    predictions = [
        prediction
        for task_result in task_results
        for prediction in task_result["predictions"]
    ]
    names = [prediction["image"] for prediction in predictions]
    expected_names = set(expected_image_names)
    if len(names) != len(set(names)):
        raise RuntimeError("Inference results contain duplicate image predictions")
    if set(names) != expected_names:
        missing = sorted(expected_names - set(names))
        unexpected = sorted(set(names) - expected_names)
        raise RuntimeError(
            f"Inference results do not match the inputs: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {prediction["image"]: prediction for prediction in predictions}


def group_batches_by_library_load(results: Iterable[dict]) -> dict[str, list[str]]:
    """Group completed batches by the library instance that processed them."""
    reuse = defaultdict(list)
    for result in results:
        reuse[result["library_load_id"]].append(result["batch"])
    return dict(reuse)


def save_contact_sheet(
    image_paths: Sequence[Path],
    predictions: dict[str, dict],
    output_path: Path | str,
    *,
    max_images: int = 24,
    columns: int = 4,
) -> int:
    """Save a dynamically sized contact sheet and return its displayed count."""
    from PIL import Image, ImageDraw, ImageFont

    selected_paths = list(image_paths[:max_images])
    if not selected_paths:
        raise ValueError("Cannot build a contact sheet without images")
    columns = max(1, min(columns, len(selected_paths)))
    rows = ceil(len(selected_paths) / columns)
    tile_width, tile_height = 240, 220
    caption_height = 52
    contact_sheet = Image.new(
        "RGB",
        (tile_width * columns, tile_height * rows),
        "white",
    )
    draw = ImageDraw.Draw(contact_sheet)
    font = ImageFont.load_default()

    for index, image_path in enumerate(selected_paths):
        with Image.open(image_path) as source_image:
            thumbnail = source_image.convert("RGB")
            thumbnail.thumbnail(
                (tile_width - 12, tile_height - caption_height - 12)
            )
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        image_x = x + (tile_width - thumbnail.width) // 2
        image_y = y + 6
        contact_sheet.paste(thumbnail, (image_x, image_y))

        prediction = predictions[image_path.name]["top_predictions"][0]
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output_path, quality=90)
    return len(selected_paths)
