"""Run MobileNet inference in-process or through either TaskVine task model."""

# This file uses ``# %%`` markers so its structure mirrors the tutorial
# notebook without duplicating the helper implementations.

# %% Cell 1: workflow configuration
import json
import os
import shutil
import time
from pathlib import Path

import cloudpickle
import mobilenet_helpers

DEFAULT_EXECUTION_MODE = "stateful-serverless"
VALID_EXECUTION_MODES = {
    "in-process",
    "python-task",
    "stateful-serverless",
}
BATCH_SIZE = 4
TOP_K = 5
LIBRARY_NAME = "mobilenetv2-inference"

MODEL_PATH = Path("data/mobilenetv2-10.onnx")
LABELS_PATH = Path("data/imagenet-synset.txt")
IMAGE_DIR = Path("data/images")
MANIFEST_PATH = Path("data/image-manifest.json")
OUTPUT_DIR = Path("outputs")


# Imported task functions must be serialized with the workflow module instead
# of being imported by name from a module that is absent in worker sandboxes.
cloudpickle.register_pickle_by_value(mobilenet_helpers)

environment_mode = os.environ.get("MOBILENET_EXECUTION_MODE")
if environment_mode is None:
    execution_mode = DEFAULT_EXECUTION_MODE
    execution_mode_source = "Python entrypoint default"
else:
    execution_mode = environment_mode.strip()
    execution_mode_source = "environment variable MOBILENET_EXECUTION_MODE"

if execution_mode not in VALID_EXECUTION_MODES:
    raise ValueError(
        f"Execution mode must be one of {sorted(VALID_EXECUTION_MODES)}; "
        f"received {execution_mode!r}"
    )

# Floability sets this variable from its --no-worker option. Checking it here
# prevents a distributed workflow from waiting forever without workers and
# prevents in-process mode from allocating unused workers.
workers_enabled = os.environ.get("FLOABILITY_WORKERS_ENABLED")
if workers_enabled not in {None, "0", "1"}:
    raise ValueError("FLOABILITY_WORKERS_ENABLED must be either '0' or '1'")
if execution_mode == "in-process" and workers_enabled == "1":
    raise RuntimeError(
        "In-process inference requires Floability's --no-worker option"
    )
if execution_mode != "in-process" and workers_enabled == "0":
    raise RuntimeError(
        f"Execution mode {execution_mode!r} requires Floability workers"
    )

print(f"[workflow] Execution mode: {execution_mode}")
print(f"[workflow] Mode source: {execution_mode_source}")


# %% Cell 2: verify the source-agnostic manifest and prepare batches
for required_path in (MODEL_PATH, LABELS_PATH):
    if not required_path.is_file():
        raise FileNotFoundError(f"Required staged input not found: {required_path}")

image_paths, manifest = mobilenet_helpers.load_and_verify_images(
    IMAGE_DIR,
    MANIFEST_PATH,
)
image_batches = mobilenet_helpers.make_image_batches(image_paths, BATCH_SIZE)
expected_image_names = [path.name for path in image_paths]
print(
    f"[data] Verified dataset {manifest['dataset_id']}@"
    f"{manifest['dataset_version']}: {len(image_paths)} images"
)
print(
    f"[data] Prepared {len(image_batches)} deterministic microbatches "
    f"with at most {BATCH_SIZE} images each"
)


# %% Cell 3: execute the selected mode
batch_root = None
library_reuse = {}

if execution_mode == "in-process":
    started_at = time.perf_counter()
    local_result = mobilenet_helpers.classify_batch_with_new_session(
        str(MODEL_PATH),
        str(LABELS_PATH),
        str(IMAGE_DIR),
        TOP_K,
    )
    local_result["batch"] = "all-images"
    local_result["worker_address"] = None
    results = [local_result]
    elapsed_seconds = time.perf_counter() - started_at
    print(f"[in-process] Classified {len(image_paths)} images without workers")
else:
    import ndcctools.taskvine as vine

    manager_name = os.environ.get("VINE_MANAGER_NAME")
    if not manager_name:
        raise RuntimeError(
            "VINE_MANAGER_NAME is not set; execute this distributed mode through "
            "Floability"
        )

    port_spec = os.environ.get("VINE_MANAGER_PORTS", "9123,9150")
    ports = [int(value.strip()) for value in port_spec.split(",") if value.strip()]
    if not ports:
        raise ValueError("VINE_MANAGER_PORTS does not contain a port")
    manager_port = ports[0] if len(ports) == 1 else [min(ports), max(ports)]

    manager = vine.Manager(port=manager_port, name=manager_name)
    manager.tune("watch-library-logfiles", 1)
    print(f"[manager] Name: {manager_name}")
    print(f"[manager] Listening on port: {manager.port}")

    batch_root, batch_paths = mobilenet_helpers.materialize_batch_directories(
        image_batches
    )
    declared_model = manager.declare_file(str(MODEL_PATH), cache=True)
    declared_labels = manager.declare_file(str(LABELS_PATH), cache=True)
    declared_batches = {
        batch_path: manager.declare_file(str(batch_path), cache=True)
        for batch_path in batch_paths
    }

    if execution_mode == "stateful-serverless":
        library = manager.create_library_from_functions(
            LIBRARY_NAME,
            mobilenet_helpers.classify_batch_with_shared_session,
            add_env=False,
            exec_mode="direct",
            library_context_info=[
                mobilenet_helpers.initialize_mobilenet_library,
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

    task_batches = {}
    started_at = time.perf_counter()
    for batch_path in batch_paths:
        if execution_mode == "python-task":
            task = vine.PythonTask(
                mobilenet_helpers.classify_batch_with_new_session,
                "model.onnx",
                "labels.txt",
                "batch",
                TOP_K,
            )
            task.add_input(declared_model, "model.onnx")
            task.add_input(declared_labels, "labels.txt")
        else:
            task = vine.FunctionCall(
                LIBRARY_NAME,
                "classify_batch_with_shared_session",
                "batch",
                TOP_K,
            )

        task.add_input(declared_batches[batch_path], "batch")
        task.set_cores(1)
        task_id = manager.submit(task)
        task_batches[task_id] = batch_path.name

    print(f"[tasks] Submitted {len(task_batches)} tasks using {execution_mode}")

    results = []
    failures = []
    while not manager.empty():
        completed = manager.wait(5)
        if not completed:
            continue
        if not completed.successful():
            failures.append((completed.id, completed.result))
            print(f"[tasks] FAILED task={completed.id} result={completed.result}")
            continue

        result = completed.output
        if execution_mode == "stateful-serverless":
            if result["function_pid"] != result["library_pid"]:
                raise RuntimeError(
                    "FunctionCall did not execute inside its persistent library process"
                )
            execution_detail = f"load_id={result['library_load_id']}"
        else:
            execution_detail = f"session={result['session_load_id']}"

        result["task_id"] = completed.id
        result["batch"] = task_batches[completed.id]
        result["worker_address"] = completed.addrport
        results.append(result)
        print(
            f"[tasks] task={completed.id} batch={result['batch']} "
            f"{execution_detail} worker={completed.addrport}"
        )

    if failures:
        raise RuntimeError(f"Inference task failures: {failures}")
    if len(results) != len(batch_paths):
        raise RuntimeError(
            f"Expected {len(batch_paths)} task results; received {len(results)}"
        )
    elapsed_seconds = time.perf_counter() - started_at

    if execution_mode == "stateful-serverless":
        library_reuse = mobilenet_helpers.group_batches_by_library_load(results)
        print(f"[stateful] Persistent library instances: {len(library_reuse)}")
        for load_id, batches in sorted(library_reuse.items()):
            print(f"[stateful]   {load_id}: {len(batches)} microbatch(es)")


# %% Cell 4: validate and save common outputs
predictions = mobilenet_helpers.predictions_by_image(
    results,
    expected_image_names,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary = {
    "dataset_id": manifest["dataset_id"],
    "dataset_version": manifest["dataset_version"],
    "execution_mode": execution_mode,
    "execution_mode_source": execution_mode_source,
    "image_count": len(image_paths),
    "batch_size": BATCH_SIZE,
    "batch_count": len(image_batches),
    "elapsed_seconds": elapsed_seconds,
    "distinct_library_loads": (
        len(library_reuse) if execution_mode == "stateful-serverless" else None
    ),
    "task_results": results,
}
summary_path = OUTPUT_DIR / f"{execution_mode}-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

contact_sheet_path = OUTPUT_DIR / f"{execution_mode}-contact-sheet.jpg"
displayed_image_count = mobilenet_helpers.save_contact_sheet(
    image_paths,
    predictions,
    contact_sheet_path,
)

if batch_root is not None:
    shutil.rmtree(batch_root)

print("=" * 72)
print("MOBILENET BATCH INFERENCE COMPLETE")
print(f"Execution mode: {execution_mode}")
print(f"Validated images: {len(predictions)}")
print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
print(f"Results: {summary_path}")
print(
    f"Contact sheet: {contact_sheet_path} "
    f"({displayed_image_count} images shown)"
)
print("=" * 72)
