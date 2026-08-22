# MobileNet Batch Inference

This backpack runs CPU-only MobileNetV2 ONNX inference over a manifest-defined
image dataset. The default `wikimedia_24` data profile contains 24 openly
licensed Wikimedia Commons images and serves as a self-contained installation
and smoke test. The images are a tutorial dataset, not an accuracy benchmark.

The notebook and Python entrypoint support three execution modes:

- `in-process` loads and runs the model without TaskVine workers;
- `python-task` distributes ordinary `PythonTask` microbatches that each load
  the model; and
- `stateful-serverless` uses persistent `LibraryTask` instances and
  `FunctionCall` tasks that can reuse a loaded ONNX Runtime session.

Set the mode with `MOBILENET_EXECUTION_MODE`. In the notebook, the first
configuration cell is used when that environment variable is absent.

## Run interactively

Install and activate Floability using the
[official installation instructions](https://floability.readthedocs.io/en/stable/getting-started/installation/),
then start the notebook and local workers for either distributed mode:

```bash
floability run \
  --backpack external-example/mobilenet-batch-inference \
  --sync-path outputs
```

Open the JupyterLab URL printed by Floability, select the execution method in
the first configuration cell, and run the notebook from top to bottom. To try
the other method, restart the kernel, change the selection, and run all cells
again.

To run entirely in the notebook process, do not start workers:

```bash
floability run \
  --backpack external-example/mobilenet-batch-inference \
  --no-worker \
  --env-vars MOBILENET_EXECUTION_MODE=in-process \
  --sync-path outputs
```

The workflow rejects `in-process` mode when workers were enabled, and rejects
a distributed mode when Floability was launched with `--no-worker`. This makes
an accidental configuration mismatch fail immediately.

## Run non-interactively

The backpack now contains both notebook and Python entrypoints. Select the
Python file explicitly for non-interactive execution. The Python entrypoint
defaults to stateful serverless execution:

```bash
floability execute \
  --backpack external-example/mobilenet-batch-inference \
  --entrypoint mobilenet-batch-inference.py \
  --sync-path outputs
```

Select another mode with `--env-vars`:

```bash
floability execute \
  --backpack external-example/mobilenet-batch-inference \
  --entrypoint mobilenet-batch-inference.py \
  --env-vars MOBILENET_EXECUTION_MODE=python-task \
  --sync-path outputs
```

For worker-free execution, combine the mode with `--no-worker`:

```bash
floability execute \
  --backpack external-example/mobilenet-batch-inference \
  --entrypoint mobilenet-batch-inference.py \
  --no-worker \
  --env-vars MOBILENET_EXECUTION_MODE=in-process \
  --sync-path outputs
```

For an HPC batch system, select the site's batch type:

```bash
floability execute \
  --backpack external-example/mobilenet-batch-inference \
  --entrypoint mobilenet-batch-inference.py \
  --batch-type condor \
  --sync-path outputs
```

```bash
floability execute \
  --backpack external-example/mobilenet-batch-inference \
  --entrypoint mobilenet-batch-inference.py \
  --batch-type slurm \
  --sync-path outputs
```

Use `--base-dir /path/with/space` when the default location does not have room
for the built and packed Conda environments. At sites that restrict incoming
worker connections, add `--manager-port-range 10000:11000`. Use
`--worker-transfer-port-range 11001:12000` when peer-to-peer transfers must
also stay within an allowed range.

## Dataset contract

Every profile must stage an image directory at `data/images` and its manifest
at `data/image-manifest.json`. Manifest schema version 1 records:

- `dataset_id`, `dataset_version`, and `image_count`;
- a flat, unique filename for every image; and
- the SHA-256 checksum of every image.

The workflow validates the manifest, sorts images by filename, and divides any
positive image count into deterministic microbatches. Therefore a future S3
profile can stage a larger directory and manifest without adding a separate
workflow code path. `BATCH_SIZE` is configurable in the first notebook cell or
near the top of the Python entrypoint.

## What the workflow produces

Each mode writes its own JSON summary and contact sheet under `outputs/`.
Summaries record the dataset version, timing, predictions, and available
process, worker, and library metadata. Contact sheets show at most the first 24
deterministically ordered images so output remains manageable for larger data
profiles. Repeated library-load IDs in stateful output demonstrate model-state
reuse when scheduling assigns multiple calls to a library instance.

Timing includes different costs in each mode, so this small tutorial run
should not be treated as a formal performance benchmark.

Image authors and licenses are listed in [IMAGE_CREDITS.md](IMAGE_CREDITS.md).
Model checksums and upstream revisions are recorded in
[MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).
