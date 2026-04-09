<p align="center">
  <img src="assets/logo.png" alt="MolmoWeb" width="100%">
</p>

<p align="center">
  <a href="https://allenai.org/papers/molmoweb">Paper</a> &nbsp;|&nbsp;
  <a href="https://allenai.org/blog/molmoweb">Blog Post</a> &nbsp;|&nbsp;
  <a href="https://molmoweb.allen.ai">Demo</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/collections/allenai/molmoweb">Models</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/collections/allenai/molmoweb-data">Data</a>
</p>

---

**MolmoWeb** is an open multimodal web agent built by [Ai2](https://allenai.org). Given a natural-language task, MolmoWeb autonomously controls a web browser -- clicking, typing, scrolling, and navigating -- to complete the task. This repository contains the agent code, inference client, evaluation benchmarks, and everything needed to reproduce the results from the paper.

## Table of Contents

- [Models](#models)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Download the Model](#1-download-the-model)
  - [Start the Model Server](#2-start-the-model-server)
  - [Test the Model](#3-test-the-model)
- [Inference Client](#inference-client)
  - [Single Query](#single-query)
  - [Batch Queries](#batch-queries)
  - [Extract Accessibility Tree](#extract-accessibility-tree)
- [Benchmarks](#benchmarks)
- [Annotation Tool](annotation/README.md)
- [Training](#training)
  - [Setup](#setup)
  - [Downloading Data](#downloading-data)
  - [Downloading Pretrained Checkpoints](#downloading-pretrained-checkpoints)
  - [SFT Training](#sft-training)
  - [Key Training Arguments](#key-training-arguments)
- [Grounding Evaluation](#grounding-evaluation)
- [License](#license)
- [TODO](#todo)

---

## Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| MolmoWeb-8B | 8B | [allenai/MolmoWeb-8B](https://huggingface.co/allenai/MolmoWeb-8B) |
| MolmoWeb-4B | 4B | [allenai/MolmoWeb-4B](https://huggingface.co/allenai/MolmoWeb-4B) |
| MolmoWeb-8B-Native | 8B | [allenai/MolmoWeb-8B-Native](https://huggingface.co/allenai/MolmoWeb-8B-Native) |
| MolmoWeb-4B-Native | 4B | [allenai/MolmoWeb-4B-Native](https://huggingface.co/allenai/MolmoWeb-4B-Native) |

The first two models (MolmoWeb-8B and MolmoWeb-4B) are Huggingface/transformers-compatible (see [example usage](https://huggingface.co/allenai/MolmoWeb-8B#quick-start) on Huggingface); and the last two (MolmoWeb-8B-Native and MolmoWeb-4B-Native) are molmo-native checkpoints. 

**Collections:**
- [All MolmoWeb Models](https://huggingface.co/collections/allenai/molmoweb)
- [MolmoWeb Training Data](https://huggingface.co/collections/allenai/molmoweb-data)

---

## Installation

Requires Python >=3.10,<3.13. We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone git@github.com:allenai/molmoweb.git
cd molmoweb
uv venv --python ">=3.10,<3.13"
uv sync

# Install Playwright browsers (needed for local browser control)
uv run playwright install
uv run playwright install --with-deps chromium
```

---

### Environment Variables

```bash
# Browserbase (required when --env_type browserbase)
export BROWSERBASE_API_KEY="your-browserbase-api-key"
export BROWSERBASE_PROJECT_ID="your-browserbase-project-id"

# Google Gemini (required for gemini_cua, gemini_axtree, and Gemini-based judges)
export GOOGLE_API_KEY="your-google-api-key"

# OpenAI (required for gpt_axtree and GPT-based judges like webvoyager)
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Quick Start

Three helper scripts in `scripts/` let you download weights, start the server, and test it end-to-end.

### 1. Download the Model

```bash
bash scripts/download_weights.sh                                  # MolmoWeb-8B (default)
bash scripts/download_weights.sh allenai/MolmoWeb-4B-Native       # MolmoWeb-4B Native
```

This downloads the weights to `./checkpoints/<model-name>`.

### 2. Start the Model Server

```bash
# default predictor type is native
bash scripts/start_server.sh ./checkpoints/MolmoWeb-4B-Native       # MolmoWeb-4B-Native
# change to HF-compatible
export PREDICTOR_TYPE="hf"
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B              # MolmoWeb-8B, port 8001
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B 8002         # custom port
```

Or configure via environment variables:

```bash
export CKPT="./checkpoints/MolmoWeb-4B-Native"   # local path to downloaded weights
export PREDICTOR_TYPE="native"             # "native" or "hf"
export NUM_PREDICTORS=1                    # number of GPU workers

bash scripts/start_server.sh
```

The server exposes a single endpoint:

```
POST http://127.0.0.1:8001/predict
{
  "prompt": "...",
  "image_base64": "..."
}
```

Wait for the server to print that the model is loaded, then test it.

### 3. Test the Model

Once the server is running, send it a screenshot of the [Ai2 careers page](https://allenai.org/careers) (included in `assets/test_screenshot.png`) and ask it to read the job titles:

```bash
uv run python scripts/test_server.py                        # default: localhost:8001
uv run python scripts/test_server.py http://myhost:8002     # custom endpoint
```

The test script sends this prompt to the model:

> Read the text on this page. What are the first four job titles listed under 'Open roles'?

You can also do it manually in a few lines of Python:

```python
import base64, requests

with open("assets/test_screenshot.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://127.0.0.1:8001/predict", json={
    "prompt": "What are the first four job titles listed under 'Open roles'?",
    "image_base64": image_b64,
})
print(resp.json())
```

---

## Inference Client

The `inference` package provides a high-level Python client that manages a browser session and runs the agent end-to-end. The client communicates with a running model server endpoint.

### Single Query

```python
from inference import MolmoWeb

client = MolmoWeb(
    endpoint="SET_UP_YOUR_ENDPOINT",
    local=True,         # True = local Chromium, False = Browserbase cloud browser
    headless=True,
) 

query = "Go to arxiv.org and find out the paper about Molmo and Pixmo."
traj = client.run(query=query, max_steps=10)

output_path = traj.save_html(query=query)
print(f"Saved to {output_path}")
```

### Follow-up Query

```python
followup_query = "Find the full author list of the paper."
traj2 = client.continue_run(query=followup_query, max_steps=10)
```

### Batch Queries

```python
queries = [
    "Go to allenai.org and find the latest research papers on top of the homepage",
    "Search for 'OLMo' on Wikipedia",
    "What is the weather in Seattle today?",
]

trajectories = client.run_batch(
    queries=queries,
    max_steps=10,
    max_workers=3,
) # Inspect the trajectory .html files default saved under inference/htmls
```

### Inference Backends

Supported backends: `fastapi` (remote HTTP endpoint), `modal` (serverless), `native` (native molmo/olmo-compatible checkpoint), `hf` (HuggingFace Transformers-compatible checkpoint).

> **vLLM support coming soon.**

### Extract Accessibility Tree

```
from inference.client import MolmoWeb

client = MolmoWeb()
axtree_str = client.get_axtree("https://allenai.org/")
print(axtree_str)
client.close()
```

---

## Benchmarks

The `benchmarks/` directory contains the unified evaluation framework. It supports five benchmarks out of the box: **WebVoyager**, **Online Mind2Web**, **DeepShop**, **WebTailBench**, and **Custom** (bring your own tasks).

The evaluation pipeline has two stages:

1. **Run** -- the agent executes tasks in a browser, producing trajectory logs.
2. **Judge** -- an LLM judge scores each trajectory for success.

### Running Evaluations

The entry point is `benchmarks/benchmarks.py`, a [Fire](https://github.com/google/python-fire) CLI with two commands: `run` and `judge`.

```bash
uv run python -m benchmarks.benchmarks run \
    --benchmark custom \
    --data_path ./demo_task.json \
    --results_dir ./results \
    --agent_type molmoweb \
    --inference_mode fastapi \
    --endpoint_or_checkpoint http://127.0.0.1:8001 \
    --max_steps 30 \
    --num_workers 1 \
    --env_type simple
```

### Judging Results

After trajectories are collected, run the judge. The `webvoyager` judge requires `OPENAI_API_KEY` to be set.

```bash
uv run python -m benchmarks.benchmarks judge \
    --benchmark custom \
    --data_path ./demo_task.json \
    --results_dir ./results \
    --judge_type webvoyager \
    --num_workers 1
```

### Synthetic Data Generation

The same evaluation framework can be used to generate synthetic training data by running other agents on tasks. Collect trajectories with any supported agent and use the resulting logs for training.

### Agents

| Agent | Description | Required Environment Variables |
|-------|-------------|-------------------------------|
| `molmoweb` | MolmoWeb multimodal agent (local model server) | None (uses `--endpoint_or_checkpoint`) |
| `gemini_cua` | Gemini computer-use agent | `GOOGLE_API_KEY` |
| `gemini_axtree` | Gemini with accessibility tree | `GOOGLE_API_KEY` |
| `gpt_axtree` | GPT with accessibility tree | `OPENAI_API_KEY` |

We welcome contributions of custom agents. To add your own, implement the agent interface in `agent/` and register the agent type in `benchmarks/evaluate.py`.

### Evaluating Other Agents on Benchmarks

You can evaluate any supported agent on any benchmark using the same code. For example, to evaluate `gemini_axtree` on Online Mind2Web with Browserbase:

```bash
uv run python -m benchmarks.benchmarks run \
    --benchmark online_mind2web \
    --results_dir ./results/om2w_gemini_axtree \
    --agent_type gemini_axtree \
    --max_steps 30 \
    --num_workers 5 \
    --env_type browserbase
```

Then judge the results:

```bash
uv run python -m benchmarks.benchmarks judge \
    --benchmark online_mind2web \
    --results_dir ./results/om2w_gemini_axtree \
    --judge_type webjudge_online_mind2web \
    --num_workers 5
```

### benchmarks.py Reference

#### `run` command

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `results_dir` | `str` | *(required)* | Output directory for trajectory logs. |
| `agent_type` | `str` | *(required)* | Agent to use: `molmoweb`, `gemini_cua`, `gemini_axtree`, or `gpt_axtree`. |
| `benchmark` | `str` | `"custom"` | Benchmark name: `custom`, `deepshop`, `webvoyager`, `online_mind2web`, or `webtailbench`. |
| `data_path` | `str` | `None` | Override the default data file path for the chosen benchmark. |
| `inference_mode` | `str` | `None` | How to connect to the model: `fastapi` (HTTP endpoint), `local` (in-process HF), `modal` (Modal serverless), or `native` (in-process OLMo). |
| `endpoint_or_checkpoint` | `str` | `None` | Either an HTTP URL (for `fastapi`/`modal`) or a local path / HF model ID (for `local`/`native`). |
| `device` | `str` | `None` | CUDA device for local inference, e.g. `cuda:0`. |
| `api_key` | `str` | `None` | API key for API-based agents (Gemini, GPT). |
| `num_workers` | `int` | `5` | Number of parallel evaluation workers. |
| `max_steps` | `int` | `30` | Maximum agent steps per episode. |
| `env_type` | `str` | `"simple"` | Browser environment: `browserbase` (requires `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID`) or `simple` (local Chromium). |

#### `judge` command

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `results_dir` | `str` | *(required)* | Directory containing trajectory logs to judge. |
| `benchmark` | `str` | `"custom"` | Benchmark name (must match what was used during `run`). |
| `data_path` | `str` | `None` | Override data file path. |
| `judge_type` | `str` | `None` | Judge implementation. Defaults to the benchmark's default judge. Options: `webvoyager` (GPT-4o), `deepshop_judge`, `webjudge_online_mind2web`. |
| `num_workers` | `int` | `30` | Number of parallel judging workers. |

See [benchmarks/README.md](benchmarks/README.md) for full documentation.

---

## Training

Training code lives in the `train/` directory. MolmoWeb training is a single-stage SFT on top of a Molmo2 pretrained checkpoint.

### Setup

Install dependencies inside the `train/` directory:

```bash
cd train
uv sync
```

Set the following environment variables before training:

```
WANDB_API_KEY=your_wandb_key
HF_ACCESS_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key   # used in some evaluations
OLMO_SHARED_FS=1                 # for multi-node jobs on a shared filesystem
OMP_NUM_THREADS=8
```

### Downloading Data

MolmoWeb training data is hosted on HuggingFace under the [MolmoWeb Data collection](https://huggingface.co/collections/allenai/molmoweb-data). Set `WEBOLMO_DATA_DIR` to where you want the data stored (defaults to `/weka/oe-training-default/webolmo/datasets`):

```bash
export WEBOLMO_DATA_DIR=/path/to/webolmo/datasets
```

Then download all datasets with:

```bash
bash scripts/download_datasets.sh
```

This downloads the following datasets from HuggingFace:

| Dataset | HuggingFace Repo | Description |
|---|---|---|
| SyntheticGround | `allenai/MolmoWeb-SyntheticGround` | Synthetic web grounding (click targets) |
| SyntheticQA | `allenai/MolmoWeb-SyntheticQA` | Synthetic screenshot QA |
| SyntheticTrajs | `allenai/MolmoWeb-SyntheticTrajs` | Gemini-generated agent trajectories |
| HumanTrajs | `allenai/MolmoWeb-HumanTrajs` | Human-annotated trajectories |
| SyntheticSkills | `allenai/MolmoWeb-SyntheticSkills` | Synthetic atomic skill demonstrations |
| HumanSkills | `allenai/MolmoWeb-HumanSkills` | Human atomic skill demonstrations |

Training also uses image pointing data from [Molmo/PixMo](https://huggingface.co/collections/allenai/pixmo-674563dc2e11d2f68e4a4901). Set `DATA_DIR` / `MOLMO_DATA_DIR` to where those are stored.

### Downloading Pretrained Checkpoints

SFT training starts from a Molmo2 pretrained checkpoint. Download one of the pretrained base checkpoints from HuggingFace:

```bash
bash scripts/download_weights.sh allenai/MolmoWeb-Pretrained-8B   # 8B base
bash scripts/download_weights.sh allenai/MolmoWeb-Pretrained-4B   # 4B base
```

This saves the checkpoint to `./checkpoints/MolmoWeb-Pretrained-8B` (or `-4B`). Set `CHECKPOINT_PATH` in `train/run_train.sh` to this path before launching training.

| Model | HuggingFace Repo |
|---|---|
| MolmoWeb-Pretrained-8B | [allenai/MolmoWeb-Pretrained-8B](https://huggingface.co/allenai/MolmoWeb-Pretrained-8B) |
| MolmoWeb-Pretrained-4B | [allenai/MolmoWeb-Pretrained-4B](https://huggingface.co/allenai/MolmoWeb-Pretrained-4B) |

### SFT Training

The entry point is `train/launch_scripts/train.py`. The first positional argument is the data mixture name and the second is the path to the starting checkpoint (a Molmo2 pretrained or SFT checkpoint).

The easiest way to launch training is via `scripts/run_train.sh`, which wraps `torchrun` with the default configuration:

```bash
bash scripts/run_train.sh
```

Key variables to configure at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_PATH` | Molmo2 4B step30000 | Starting checkpoint path |
| `MIXTURE` | `molmoweb` | Training data mixture (`molmoweb` or `debug`) |
| `NUM_GPUS` | `8` | GPUs per node |
| `GLOBAL_BATCH_SIZE` | `64` | Total batch size across all GPUs |
| `DEVICE_BATCH_SIZE` | `2` | Per-GPU batch size |
| `SEQ_LEN` | `10240` | Sequence length |
| `DURATION` | `500` | Number of training steps |
| `SAVE_INTERVAL` | `100` | Checkpoint save frequency (steps) |

To launch a debug run directly:

```bash
cd train
torchrun -m --nproc-per-node 1 \
  launch_scripts.train debug debug \
  --save_folder=dbg \
  --device_batch_size 1 \
  --duration 10 \
  --global_batch_size 2
```

### Key Training Arguments

| Argument | Description |
|---|---|
| `mixture` | Data mixture: `molmoweb`, `hero`, or `debug` |
| `checkpoint` | Path to the starting checkpoint (or `debug`) |
| `--seq_len` | Sequence length (default: `auto`) |
| `--global_batch_size` | Total batch size |
| `--device_batch_size` | Per-device batch size |
| `--duration` | Number of training steps |
| `--save_interval` | Checkpoint save frequency |
| `--connector_lr` | LR for the vision-language connector (default: `5e-6`) |
| `--llm_lr` | LR for the LLM backbone (default: `1e-5`) |
| `--vit_lr` | LR for the vision encoder (default: `5e-6`) |
| `--warmup_steps` | LR warmup steps (default: `200`) |
| `--num_checkpoints_to_keep` | How many recent checkpoints to retain |

---

## Grounding Evaluation

MolmoWeb can be evaluated on grounding benchmarks to measure how accurately the model predicts click coordinates for UI elements. The entry point is `launch_scripts.eval`, run via `torchrun` from inside the `train/` directory.

### Supported Benchmarks

| Benchmark | Task name |
|---|---|
| [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) | `screenspot` |
| [ScreenSpot-v2](https://huggingface.co/datasets/likaixin/ScreenSpot-v2) | `screenspot_v2` |
| [WebClick](https://huggingface.co/datasets/allenai/MolmoWeb-SyntheticGround) | `webclick` |
| [GroundUI-1K](https://huggingface.co/datasets/BigAction/GroundUI-1K) | `groundui_1k` |

### Running Grounding Eval

The easiest way is via `train/run_ground_eval.sh`. Configure these variables at the top of the script:

| Variable | Description |
|---|---|
| `CHECKPOINT_PATH` | Path to the checkpoint to evaluate |
| `MIXTURE` | Comma-separated list of `task:split` pairs (e.g. `screenspot:test,screenspot_v2:test`) |
| `NUM_GPUS` | Number of GPUs to use |
| `DEVICE_BATCH_SIZE` | Per-GPU batch size |
| `SAVE_FOLDER` | Directory to write results |

Then run:

```bash
cd train
bash run_ground_eval.sh
```

Required environment variables:

```bash
export DATA_DIR=/path/to/molmo/data          # Molmo/PixMo image pointing data
export MOLMO_DATA_DIR=$DATA_DIR
export WEBOLMO_DATA_DIR=/path/to/webolmo/datasets
export WEBOLMO_DATASET_VERSION=<dataset_version>
```

To run directly with `torchrun`:

```bash
cd train
DATA_DIR=... MOLMO_DATA_DIR=... WEBOLMO_DATA_DIR=... WEBOLMO_DATASET_VERSION=... \
torchrun --nproc-per-node 1 -m \
  launch_scripts.eval /path/to/checkpoint \
  screenspot:test,screenspot_v2:test \
  --save_dir=./results/eval_run \
  --device_batch_size 2 \
  --include_image
```

### Key Eval Arguments

| Argument | Default | Description |
|---|---|---|
| `checkpoint` | *(required)* | Path to the checkpoint to evaluate |
| `tasks` | *(required)* | Comma-separated `task:split` pairs (e.g. `screenspot:test`) |
| `--save_dir` | checkpoint dir | Directory to write result files |
| `--device_batch_size` | `4` | Per-GPU batch size |
| `--include_image` | `false` | Save screenshots alongside predictions |
| `--max_examples` | `None` | Limit evaluation to N examples |
| `--overwrite` | `false` | Rerun even if cached metrics exist |
| `--seq_len` | `None` | Override sequence length |
| `--num_workers` | `2` | Data loading workers |

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## TODO

- [x] Inference
- [x] Eval
- [x] Training
- [x] Annotation Tool
