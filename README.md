# MobileNet Batch Inference

This backpack runs CPU-only MobileNetV2 ONNX inference over a
manifest-defined image dataset. Its default `wikimedia_24` profile contains 24
openly licensed Wikimedia Commons images, making the backpack a small
installation and execution test rather than an accuracy benchmark.

The backpack intentionally provides multiple workflow entrypoints so the two
TaskVine execution models can be inspected separately:

- `mobilenet-python-task.ipynb` submits ordinary TaskVine `PythonTask` tasks.
  Each image microbatch creates a new ONNX Runtime session.
- `mobilenet-serverless-taskvine.ipynb` installs a persistent TaskVine Function
  Library and submits Function Calls. A library process can reuse its loaded
  model across multiple microbatches.
- `mobilenet-batch-inference.py` is the equivalent headless Python entrypoint.
  It supports `python-task`, `stateful-serverless`, and worker-free
  `in-process` modes.

Because more than one eligible workflow exists, select an entrypoint explicitly
with `--entrypoint`.

## Run a notebook interactively

Install and activate Floability using the
[official installation instructions](https://floability.readthedocs.io/en/stable/getting-started/installation/),
clone this repository, and enter its directory.

Run the ordinary PythonTask notebook:

```bash
floability run --backpack . \
  --entrypoint mobilenet-python-task.ipynb \
  --sync-path outputs
```

Or run the stateful Function Library notebook:

```bash
floability run --backpack . \
  --entrypoint mobilenet-serverless-taskvine.ipynb \
  --sync-path outputs
```

Open the JupyterLab URL printed by Floability and run the selected notebook from
top to bottom. Floability starts the local TaskVine workers unless a batch type
is selected.

## Run non-interactively

Either notebook can be executed without opening a browser:

```bash
floability execute --backpack . \
  --entrypoint mobilenet-python-task.ipynb \
  --sync-path outputs
```

```bash
floability execute --backpack . \
  --entrypoint mobilenet-serverless-taskvine.ipynb \
  --sync-path outputs
```

The Python entrypoint defaults to stateful serverless execution:

```bash
floability execute --backpack . \
  --entrypoint mobilenet-batch-inference.py \
  --sync-path outputs
```

Select ordinary PythonTask execution explicitly with:

```bash
floability execute --backpack . \
  --entrypoint mobilenet-batch-inference.py \
  --env-vars MOBILENET_EXECUTION_MODE=python-task \
  --sync-path outputs
```

The Python entrypoint can also provide a worker-free baseline:

```bash
floability execute --backpack . \
  --entrypoint mobilenet-batch-inference.py \
  --no-worker \
  --env-vars MOBILENET_EXECUTION_MODE=in-process \
  --sync-path outputs
```

For an HPC batch system, add `--batch-type condor` or `--batch-type slurm`
to a distributed command.

Use `--base-dir /path/with/space` when the default location does not have enough
space for built and packed Conda environments. If a site restricts incoming
worker connections, use `--manager-ports 10000:11000`. Use
`--worker-transfer-ports 11001:12000` when peer-to-peer transfers must also stay
within an allowed range.

## Dataset contract

Every data profile must stage an image directory at `data/images` and a
manifest at `data/image-manifest.json`. Manifest schema version 1 records:

- `dataset_id`, `dataset_version`, and `image_count`;
- a flat, unique filename for every image; and
- the SHA-256 checksum of every image.

The workflow validates the manifest, sorts images by filename, and creates
deterministic microbatches. A future data profile can therefore stage a larger
image directory and manifest without requiring a separate workflow code path.
`BATCH_SIZE` is configurable near the beginning of each entrypoint.

## Outputs

Each execution model writes a JSON summary and contact sheet under `outputs/`:

- PythonTask: `python-task-summary.json` and
  `python-task-contact-sheet.jpg`
- Function Library: `stateful-serverless-summary.json` and
  `stateful-serverless-contact-sheet.jpg`
- in-process baseline: `in-process-summary.json` and
  `in-process-contact-sheet.jpg`

Summaries include dataset identity, timing, predictions, and available process,
worker, and library metadata. Repeated library-load IDs in the stateful summary
show that multiple calls reused the same loaded model. The contact sheets show
at most the first 24 deterministically ordered images.

Image authors and licenses are listed in [IMAGE_CREDITS.md](IMAGE_CREDITS.md).
Model checksums and upstream revisions are recorded in
[MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).
