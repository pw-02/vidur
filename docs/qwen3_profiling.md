# Qwen3 profiling and importing GPU measurements

The fork now registers `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B`. Both use BF16,
headwise Q/K RMSNorm, no QKV bias and the official dense-model dimensions.
The MLP/reference projection profiler includes Q/K normalization before RoPE.
For compatibility, `time_stats.attn_rope.*` measures **Q/K normalization plus RoPE**
for these models. Vidur's existing predictor consumes that combined cost once;
its existing metric label remains "RoPE". Attention-kernel profiling measures the
attention kernels separately and does not add a second Q/K normalization cost.

## 1. Prepare a GPU profiling environment

Use a separate environment from your production serving installation. Install the
Sarathi `vidur` branch and this fork as described in `docs/profiling.md`. The profiler
imports Sarathi kernels, torch and Ray; installing Vidur's CPU requirements alone
is insufficient. From this repository root, install with `pip install -e .` in that
environment. No actual checkpoint weights are downloaded by the reference compute
profiler: it constructs dummy-weight operators using the model configuration.
This is operator profiling, not a benchmark of your actual vLLM service.

## 2. First run a small GPU profiling sweep

Inside a GPU allocation, with one GPU visible:

```bash
bash examples/host_churn/profile_qwen3.sh
```

Defaults: Qwen3-8B, TP1, BF16, up to 4096 tokens and attention batch size 8.
To select Qwen3-14B or increase ranges:

```bash
QWEN_PROFILE_MODEL=Qwen/Qwen3-14B \
QWEN_PROFILE_MAX_TOKENS=33000 \
QWEN_PROFILE_MAX_BATCH=32 \
bash examples/host_churn/profile_qwen3.sh
```

Start with the small sweep to establish dependency/kernel compatibility. Long
prompt and large batch sweeps cost substantially more; memory filtering may omit
combinations. Ensure the ranges include the shapes you actually need before using
results for predictions. The host-churn extension initially supports TP1/PP1
independent workers; profiling script defaults follow that setup.

The script produces **CSV files**, not a field to paste into the simulator:

```
profiling_outputs/mlp/<timestamp>/Qwen/Qwen3-8B/mlp.csv
profiling_outputs/attention/<timestamp>/Qwen/Qwen3-8B/attention.csv
```

The two timestamps will usually differ. Keep the profiler config and your GPU,
software/kernel version and deployment dtype alongside those files for provenance.

## 3. Import the measured CSV files

For the GPU compute device name `a100`, copy the two measured files to:

```bash
mkdir -p data/profiling/compute/a100/Qwen/Qwen3-8B
cp /path/to/measured/mlp.csv data/profiling/compute/a100/Qwen/Qwen3-8B/mlp.csv
cp /path/to/measured/attention.csv data/profiling/compute/a100/Qwen/Qwen3-8B/attention.csv
```

Use the analogous model directory for Qwen3-14B. Do not relabel another model's
profiles as Qwen3. New CSV metadata records BF16 and, for compute profiles, that
Q/K normalization was included. Import rejects Qwen3 files missing those markers.
Existing legacy-model CSV files continue to work.

Run your ordinary host-churn experiment command with:

```bash
--replica_config_model_name Qwen/Qwen3-8B \
--replica_config_device a100 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 33000 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 33000 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 32
```

Use ranges supported by your measurements, not these example values blindly.
Match the trace token limit as well. Prediction models train from the imported
CSV files during initialization; profiles are not themselves latency constants.
CPU overhead, KV memory capacity, batch limits and runtime scheduler behaviour
remain separate calibration choices. Vidur's built-in `a100` SKU assumes 80GB:
match actual GPU memory before trusting capacity/queue predictions on another SKU.
TP1/PP1 does not introduce cross-worker collectives, but initialization may still
read existing configured network files.

## Validation status

CPU tests check both model configurations, per-head/GQA shape handling, extra norm
parameter accounting and profile metadata acceptance/rejection. Existing lifecycle
and reporting tests still run. **GPU kernels and GPU-generated timings have not
been executed in this workspace.** Your first small GPU run is needed to validate
the Sarathi/PyTorch/CUDA combination. Then compare single-worker and multi-worker
no-removal predictions with your real vLLM runs before interpreting churn results.
These changes do not simulate cross-request prefix caching or LMCache persistence.
