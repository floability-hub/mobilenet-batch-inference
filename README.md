# MobileNet Batch Inference

This backpack classifies 24 openly licensed Wikimedia Commons images with a
CPU-only MobileNetV2 ONNX model. It demonstrates the same inference workload
in two TaskVine execution modes: ordinary `PythonTask` tasks that load the
model for every microbatch, and stateful `FunctionCall` tasks that reuse one
loaded ONNX Runtime session per persistent worker library. The images are a
small tutorial dataset, not an accuracy benchmark.

The interactive notebook lets the user select either ordinary `PythonTask`
execution or TaskVine's Stateful Serverless Computing model. The separate
Python entrypoint runs both methods automatically and verifies that they
produce matching results.

## Run interactively

Install and activate Floability using the
[official installation instructions](https://floability.readthedocs.io/en/stable/getting-started/installation/),
then start the notebook and local workers:

```bash
floability run \
  --backpack external-example/mobilenet-batch-inference \
  --sync-path outputs
```

Open the JupyterLab URL printed by Floability, select the execution method in
the first configuration cell, and run the notebook from top to bottom. To try
the other method, restart the kernel, change the selection, and run all cells
again.

## Run the automated Python comparison

The backpack now contains both notebook and Python entrypoints. Select the
Python file explicitly for non-interactive execution:

```bash
floability execute \
  --backpack external-example/mobilenet-batch-inference \
  --entrypoint mobilenet-batch-inference.py \
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

## What the workflow does

The default profile stages 24 backpack images plus pinned, checksummed model
and label files. The manager groups the images into six four-image
microbatches, then runs both paths over the same inputs:

1. six `PythonTask` tasks each construct their own ONNX Runtime session;
2. TaskVine installs a persistent library with one session per instance;
3. six `FunctionCall` tasks reuse those sessions; and
4. the manager verifies that both paths produced the same top-five results.

`outputs/inference-summary.json` records timings, predictions, worker
addresses, process IDs, and library-load IDs. The generated
`outputs/prediction-contact-sheet.jpg` shows all images with their top
prediction. Repeated library-load IDs in the console prove state reuse.
Timing includes TaskVine scheduling, file caching, and the order in which the
two phases run, so this small tutorial run should not be treated as a formal
performance benchmark. Per-task and per-call timings are included separately
to make those effects visible.

Image authors and licenses are listed in [IMAGE_CREDITS.md](IMAGE_CREDITS.md).
Model checksums and upstream revisions are recorded in
[MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).
