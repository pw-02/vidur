import atexit
import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidur.config import (
    ClusterConfig,
    MetricsConfig,
    SarathiSchedulerConfig,
    SimulationConfig,
    VllmSchedulerConfig,
)
from vidur.entities import Batch, ExecutionTime, Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.simulator import Simulator


class Predictor:
    def get_execution_time(self, batch, stage):
        kwargs = {name: 0.0 for name in inspect.signature(ExecutionTime).parameters}
        kwargs["num_layers_per_pipeline_stage"] = 1
        kwargs["attention_prefill_execution_time"] = 1000.0
        return ExecutionTime(**kwargs)


class Metrics:
    def __init__(self):
        self.completed = Counter()

    def on_batch_end(self, time, batch, replica, memory):
        self.completed.update(r.id for r in batch.requests if r.completed)

    def __getattr__(self, name):
        return lambda *args: None


def run(
    tmp_path,
    monkeypatch,
    schedule=None,
    arrivals=(0, 0, 0.1),
    workers=2,
    sarathi=False,
    native_metrics=False,
):
    Request._id = Replica._id = Batch._id = -1
    requests = [Request(t, 8, 4) for t in arrivals]
    monkeypatch.setattr(
        ExecutionTimePredictorRegistry, "get", lambda *a, **k: Predictor()
    )
    monkeypatch.setattr(
        RequestGeneratorRegistry,
        "get",
        lambda *a, **k: SimpleNamespace(generate=lambda: requests),
    )
    metrics = Metrics()
    monkeypatch.setattr(
        "vidur.simulator.MetricsStore",
        MetricsStore if native_metrics else lambda *a: metrics,
    )
    config = SimulationConfig(
        cluster_config=ClusterConfig(
            num_replicas=workers,
            replica_scheduler_config=(
                SarathiSchedulerConfig(chunk_size=4, num_blocks=100, batch_size_cap=8)
                if sarathi
                else VllmSchedulerConfig(num_blocks=100, batch_size_cap=8)
            ),
        ),
        metrics_config=MetricsConfig(output_dir=str(tmp_path)),
    )
    if schedule is not None:
        path = tmp_path / "schedule.json"
        path.write_text(json.dumps(schedule))
        config.host_churn_file = str(path)
    simulator = Simulator(config)
    atexit.unregister(simulator._write_output)
    simulator.run()
    return simulator, requests, metrics


def test_no_churn_matches_baseline(tmp_path, monkeypatch):
    a, ra, _ = run(tmp_path, monkeypatch)
    b, rb, _ = run(tmp_path, monkeypatch, {"workers_per_host": 1})
    assert [r._completed_at for r in ra] == [r._completed_at for r in rb]
    assert [r._execution_time for r in ra] == [r._execution_time for r in rb]


@pytest.mark.parametrize("sarathi", [False, True])
@pytest.mark.parametrize("mode", ["restart", "ideal"])
def test_loss_and_retry(tmp_path, monkeypatch, mode, sarathi):
    sim, requests, metrics = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 1,
            "recovery": mode,
            "retry_delay_s": 0.2,
            "events": [{"time_s": 1.5, "action": "remove", "host": "h0"}],
        },
        sarathi=sarathi,
    )
    assert all(r.completed for r in requests)
    assert all(metrics.completed[r.id] == 1 for r in requests)
    assert sim._churn.cancelled_events > 0
    assert sim.scheduler.get_replica_scheduler(0).is_empty()
    interrupted = [
        a
        for rows in sim._churn.records.values()
        for a in rows
        if a["status"] == "interrupted"
    ]
    assert interrupted and sim._churn.records[0][0]["execution_s"] == (
        1.5 if sarathi else 1.0
    )


def test_all_offline_and_startup(tmp_path, monkeypatch):
    sim, requests, metrics = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 2,
            "events": [
                {"time_s": 0.5, "action": "remove", "host": "h0"},
                {"time_s": 2, "action": "start", "host": "h0", "startup_delay_s": 3},
            ],
        },
    )
    assert all(r.completed for r in requests)
    assert all(rows[-1]["dispatch_s"] >= 5 for rows in sim._churn.records.values())
    assert all(metrics.completed[r.id] == 1 for r in requests)


def test_stale_ready_is_invalidated(tmp_path, monkeypatch):
    sim, requests, _ = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 1,
            "initially_offline": ["h0"],
            "events": [
                {"time_s": 0, "action": "start", "host": "h0", "startup_delay_s": 3},
                {"time_s": 1, "action": "remove", "host": "h0"},
            ],
        },
    )
    assert 0 not in sim._churn.ready
    assert all(a["replica"] == 1 for rows in sim._churn.records.values() for a in rows)


def test_same_time_removal_wins(tmp_path, monkeypatch):
    sim, requests, _ = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 1,
            "events": [{"time_s": 1, "action": "remove", "host": "h0"}],
        },
    )
    assert sim._churn.records[0][0]["processed_tokens"] == 0


def test_invalid_group_size(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="workers_per_host"):
        run(tmp_path, monkeypatch, {"workers_per_host": 3})


def test_no_surviving_capacity(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="Unfinished requests"):
        run(
            tmp_path,
            monkeypatch,
            {
                "workers_per_host": 2,
                "events": [{"time_s": 0.5, "action": "remove", "host": "h0"}],
            },
        )


def test_ideal_avoids_recomputation(tmp_path, monkeypatch):
    schedule = {
        "workers_per_host": 1,
        "events": [{"time_s": 2.5, "action": "remove", "host": "h0"}],
    }
    restart, rr, _ = run(
        tmp_path, monkeypatch, {**schedule, "recovery": "restart"}, arrivals=(0,)
    )
    ideal, ri, _ = run(
        tmp_path, monkeypatch, {**schedule, "recovery": "ideal"}, arrivals=(0,)
    )
    assert ri[0]._completed_at < rr[0]._completed_at
    assert ri[0].num_processed_tokens == rr[0].num_processed_tokens == 12
    assert ideal._churn.records[0][0]["execution_s"] == 2.5


def test_explicit_unequal_hosts(tmp_path, monkeypatch):
    sim, requests, _ = run(
        tmp_path,
        monkeypatch,
        {
            "hosts": {"a": [0], "b": [1, 2]},
            "events": [{"time_s": 0.5, "action": "remove", "host": "b"}],
        },
        workers=3,
    )
    assert sim._churn.ready == {0}
    assert all(r.completed for r in requests)


def test_initially_offline_first_schedule(tmp_path, monkeypatch):
    sim, requests, _ = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 2,
            "initially_offline": ["h0"],
            "events": [{"time_s": 2, "action": "start", "host": "h0"}],
        },
    )
    assert all(r._scheduled_at == 2 for r in requests)


def test_invalid_delay(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="finite"):
        run(tmp_path, monkeypatch, {"retry_delay_s": -1})


def test_replacement_rejects_old_completion(tmp_path, monkeypatch):
    sim, requests, metrics = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 2,
            "events": [
                {"time_s": 0.5, "action": "remove", "host": "h0"},
                {"time_s": 0.6, "action": "start", "host": "h0"},
            ],
        },
        arrivals=(0,),
    )
    assert requests[0]._completed_at >= 4.6
    assert metrics.completed[0] == 1
    assert sim._churn.cancelled_events > 0


def test_partial_prefill_recovery(tmp_path, monkeypatch):
    schedule = {
        "workers_per_host": 1,
        "events": [{"time_s": 1.5, "action": "remove", "host": "h0"}],
    }
    _, restart, _ = run(
        tmp_path,
        monkeypatch,
        {**schedule, "recovery": "restart"},
        arrivals=(0,),
        sarathi=True,
    )
    _, ideal, _ = run(
        tmp_path,
        monkeypatch,
        {**schedule, "recovery": "ideal"},
        arrivals=(0,),
        sarathi=True,
    )
    assert ideal[0]._completed_at < restart[0]._completed_at
    assert ideal[0].num_processed_tokens == 12


def test_waiting_retry_records_first_execution(tmp_path, monkeypatch):
    sim, requests, _ = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 1,
            "events": [{"time_s": 0.5, "action": "remove", "host": "h0"}],
        },
        arrivals=(0, 0, 0.1),
    )
    attempts = sim._churn.records[2]
    assert "first_execution_s" not in attempts[0]
    assert requests[2]._scheduled_at == attempts[1]["first_execution_s"]


def test_native_metrics_with_churn(tmp_path, monkeypatch):
    sim, requests, _ = run(
        tmp_path,
        monkeypatch,
        {
            "workers_per_host": 1,
            "events": [{"time_s": 0.5, "action": "remove", "host": "h0"}],
        },
        native_metrics=True,
    )
    assert all(r.completed for r in requests)
    assert (Path(sim._config.metrics_config.output_dir) / "churn_summary.json").exists()
