# Adding models in this fork

The older `profiling.md` guide describes YAML files under `data/model_configs`.
This checkout actually resolves both runtime and profiling models through
`BaseModelConfig.create_from_name()` in `vidur/config/model_config.py`; there is
no `data/model_configs` directory. The following describes the current code.

1. Add a dataclass subclass of `BaseModelConfig` or a compatible model family in
   `vidur/config/model_config.py`, implementing `get_name()` with the exact model
   identifier. Classes in that module are discovered through subclasses. Set
   layers, Q/KV heads, hidden and MLP dimensions, context limit, vocabulary,
   activation, normalization, bias, RoPE and other architectural flags from the
   checkpoint's configuration. A name/dimensions entry enables selection; it does
   not establish correct performance modelling.
2. Check the reference operators in `vidur/profiling/mlp/mlp_impl.py` and the
   attention profiler against the model. Add missing operations and their costs
   when the architecture differs. Changing dimensions alone cannot cover new
   mechanisms.
3. In a separate GPU profiling environment, install the documented Sarathi
   `vidur` branch dependencies. Profile compute on the hardware used for serving.
   The profiler supports one GPU; choose TP1 for independent single-GPU workers.
   Once registered and architecturally supported, example commands are:

   ```bash
   python vidur/profiling/mlp/main.py --models Qwen/Qwen3-8B \
     --num_gpus 1 --num_tensor_parallel_workers 1 --max_tokens 33000
   python vidur/profiling/attention/main.py --models Qwen/Qwen3-8B \
     --num_gpus 1 --num_tensor_parallel_workers 1 \
     --max_model_len 33000 --max_seq_len 33000
   ```

   The fork now includes Qwen3 model/operator support. These commands require
   the separate GPU profiling dependencies and still need a GPU smoke run. Check profiler memory limits and
   its batching ranges; a complete 33k sweep can be expensive.
4. Place generated `mlp.csv` and `attention.csv` under
   `data/profiling/compute/<device>/<model-id>/`. Select the model and device with
   `--replica_config_model_name` and `--replica_config_device`. If the device is new,
   add its SKU in `vidur/config/device_sku_config.py`. Predictor paths can also be
   supplied explicitly. Its prefill, batch and total-token prediction limits must
   cover the workload; increasing trace limits alone does not increase them.
5. Network collectives are relevant to TP/PP rather than the host's number of
   independent workers. TP1/PP1 needs no new inter-worker collective measurements,
   although existing predictor initialization may still read configured network
   profile files. CPU scheduling/sampling costs are another calibration layer;
   they are not the same as GPU compute timing.
6. Validate one-worker and multiple-worker no-churn runs against real vLLM
   measurements across arrival rates before validating removal/retry scenarios.
   Compare execution, queueing and latency distributions, not just one mean.

## Qwen3-8B specifically

The published checkpoint config has 36 layers, 4096 hidden dimension, 12288 MLP
intermediate dimension, 32 Q heads, 8 KV heads, head dimension 128, vocabulary
151936, RoPE theta 1000000 and maximum positions 40960. It has no attention bias.
Check the exact checkpoint you use rather than assuming all Qwen generations match.

Qwen3 also normalizes Q and K per head. This fork now registers both Qwen3-8B
and Qwen3-14B and includes those operations in the combined Q/K-normalization-plus-
RoPE profiling time. Profilers use BF16 for these models. GPU timings still need
to be collected and validated. See [qwen3_profiling.md](qwen3_profiling.md) for the
runnable profiling script, CSV import procedure and validation limits.

Primary references:
- https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
