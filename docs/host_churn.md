# Controlled host churn

This optional extension runs inside Vidur, independently of MalleServe's real
worker launch and teardown code. Execution timings still come from Vidur's
predictor. It does not run vLLM or your actual coordinator.

## Run

Install Vidur's normal requirements and pytest for tests. Add these flags to an
otherwise working Vidur command (including its model, hardware, trace and
predictor configuration):

```bash
python -m vidur.main \
  --cluster_config_num_replicas 8 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_tensor_parallel_size 1 \
  --global_scheduler_config_type round_robin \
  --replica_scheduler_config_type vllm \
  --host_churn_file examples/host_churn/removal.json
```

This is a configuration example, not a calibrated Polaris/Qwen3 command. Supply
profiling data/model configuration appropriate to your deployment. Existing
Vidur model support is unchanged. To obtain the baseline with identical generated
request IDs and arrivals, run a separate process with the same seed and workload
and `examples/host_churn/baseline.json`. Use an empty events list (rather than
omitting the schedule) to produce the additional comparison reports.

## Schedule

`workers_per_host` is configurable and defaults to 1. It must divide the total
replica count. Sorted replica IDs are assigned consecutively to h0, h1, etc.
The example's value 4 is only an example. Alternatively supply explicit hosts:

```json
{
  "hosts": {"small": [0], "large": [1, 2, 3]},
  "initially_offline": ["large"],
  "recovery": "restart",
  "events": [
    {"time_s": 20, "action": "start", "host": "large", "startup_delay_s": 10},
    {"time_s": 50, "action": "remove", "host": "small"}
  ]
}
```

Hosts must partition every configured replica exactly once. Predeclare replacement
capacity in `num_replicas` and mark it initially offline; online/offline events
reuse these slots with new generations. `start` also clears the host's old state
if still running, then makes it available after startup. A later removal or start
invalidates an older pending readiness event. No advance warning is supplied to
routing. `detection_delay_s` plus `retry_delay_s` delays redispatch, not the physical
capacity loss. Requests wait in the global queue if no workers are ready; permanent
loss of all capacity with unfinished requests produces a report and a clear error.

## Recovery modes

- `restart`: resubmit the original prompt and original output-token target. The
  logical request keeps its ID and arrival time. Previously generated output is
  discarded. Local memory-pressure restarts continue to use Vidur's own semantics.
- `ideal`: hypothetical instantaneous recovery of KV and token progress from
  completed iterations. Work in an unfinished iteration is lost. The recovered
  context still occupies worker KV memory, and remaining computation and queueing
  still cost time. This is a diagnostic optimistic reference, not LMCache recovery
  or a measured policy. Neither mode restores cross-request prefix caches.

Use the same churn schedule in both modes to explore the benefit of avoiding
recomputation. This contrast includes downstream queue effects; it is not an
additive causal split of latency into capacity and retries. It assumes free KV
transfer for the ideal case. Calibrate no-removal behaviour and at least one real
removal before drawing numerical conclusions.

## Outputs

Normal Vidur outputs remain. Additional files in the timestamped output directory:

- `churn_requests.jsonl`: arrival, completion, total latency, recovery mode and
  attempts with dispatch, first execution, worker waiting, interruption/completion
  time, retry wait from interruption until redispatch, and execution spent. Queue time measures waiting for the first iteration
  of an attempt, not every between-iteration scheduling gap. Execution sums across
  requests are request exposure, not aggregate GPU busy time for batched work.
- `churn_summary.json`: schedule, resolved host map, lifecycle events, cancelled
  event count and interrupted-attempt count. An interrupted dispatched request
  may have been waiting rather than executing; inspect `first_execution_s`.

Unfinished stages contribute elapsed execution but no completed tokens. Partial
model-only time is not guessed. Vidur's native stage metrics omit cancelled stages;
use the churn report for failed-attempt execution accounting. At identical times,
lifecycle events precede worker events; multiple lifecycle events follow input order.

```bash
python -m vidur.host_churn.compare \
  R01=outputs/removal/churn_requests.jsonl \
  R02=outputs/baseline/churn_requests.jsonl \
  R03=outputs/ideal/churn_requests.jsonl --baseline R02 --after 90.72
python -m pytest tests/host_churn -q
```

The comparison checks IDs and arrival schedules, prints a run index, and selects
identical scheduled-arrival cohorts. It separately compares interrupted requests
against those same request IDs in the baseline, including requests that arrived
before removal. Keep all workload/model settings identical;
the comparison also verifies original prompt/output token lengths. Inspect
completion counts before interpreting completed-request latency means.

## Scope and validation

Initially supported: pipeline size 1, round-robin global routing, vLLM or Sarathi
replica scheduling. Other combinations fail explicitly. A host contains independent
replicas; multi-worker pipeline host failure is not implemented. This extension
adds no LMCache persistence, prefix prefetching or failure-aware routing policy.
Tests use real Vidur events/schedulers and deterministic timing, so they validate
lifecycle semantics rather than the accuracy of hardware performance predictions.

## Convert MalleServe request summaries

```bash
python -m vidur.host_churn.convert_requests /path/to/requests.jsonl workload.csv
```

This small standard-library-only converter preserves scheduled arrival offsets,
orders ties by trace index, prefers actual input-token counts and falls back to
recorded input tokens. It uses requested output tokens by default, includes failed
logical requests, ignores warmup records and rejects duplicate request IDs instead
of counting attempts as new arrivals. Rows without phase are treated as replay.
`--decode actual` explicitly uses measured output length; that requires an actual
completion count for every row. Convert one reference run and reuse its CSV across
simulated scenarios. The printed required token limit includes prompt plus output;
set the trace limit and predictor ranges accordingly. The extra `request_id` column
is retained as a mapping; Vidur uses numeric request IDs in CSV row order.

See [adding_models_current.md](adding_models_current.md) for the current model and
profiling path, and [qwen3_profiling.md](qwen3_profiling.md) for Qwen3 profiling and CSV import.

### Comparison report and figures

From the repository root, run:

```bash
python -m vidur.host_churn.compare
```

This selects the latest timestamped subfolder under each of
`simulator_output/baseline`, `simulator_output/removal`, and
`simulator_output/ideal`, assigning R01, R02, and R03 respectively. It prints
and saves `churn_comparison/report.md`, plus two PNG figures when Matplotlib
is installed. No Chrome or Kaleido is required. Check the run index when using
latest-run discovery; it does not guarantee the selected runs used identical
settings. Use explicit selections for reproducible comparisons:

```bash
python -m vidur.host_churn.compare --runs \
  R01=simulator_output/baseline/<timestamp> \
  R02=simulator_output/removal/<timestamp> \
  R03=simulator_output/ideal/<timestamp> \
  --baseline R01 --after 90.72 --output-dir churn_comparison
```

Positional `R01=path` arguments also work. Paths may be run folders or JSONL
files. The time used to divide requests defaults to the earliest observed removal in a non-baseline
run (zero if no summary records a removal); use `--after` to override it.
Use `--root` for a different discovery root, `--bin-seconds 10` for arrival
intervals, or `--no-plots` for a standard-library-only report.

The report includes completion counts, mean/p95 latency, summed worker wait
per request, comparisons before and after removal using the same requests, requests that
were not interrupted in any run, and interrupted requests compared with the
baseline. It also compares retrying from the beginning with resuming completed
progress. Tables comparing two runs only include requests completed in both. Figures
use a common cohort completed in every run, and short IDs resolve through the
run index. Removing interrupted requests from a table does not remove their
congestion effects. The `restart` option retries interrupted requests from the beginning. The
`ideal` option resumes completed progress on the next worker, assuming that
progress is available without modelling the cost of saving or transferring it.
An unfinished iteration is still lost, and both options still include retry
waiting and lost serving capacity. The report displays these explanations
rather than the configuration names alone. Default
profiles do not establish calibration to Polaris or Qwen.
