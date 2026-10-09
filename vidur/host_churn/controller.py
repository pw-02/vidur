"""Controlled host loss; timing prediction remains owned by Vidur."""

import json
import math
from pathlib import Path
from types import MethodType


def seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Times and delays must be finite and nonnegative")
    return value


class ChurnEvent:
    """Lifecycle events precede worker events at the same timestamp."""

    _counter = 0

    def __init__(self, time, action, host=None, request=None, generation=None):
        ChurnEvent._counter += 1
        self._time = time
        self.action, self.host = action, host
        self.request, self.generation = request, generation
        self._priority_number = (time, -1, ChurnEvent._counter)

    @property
    def time(self):
        return self._time

    def handle_event(self, scheduler, metrics_store):
        return scheduler._host_churn.handle(self)

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.action,
            "host": self.host,
            "request": self.request.id if self.request else None,
        }

    def to_chrome_trace(self):
        return None


class HostChurn:
    def __init__(self, simulator, filename):
        self.simulator = simulator
        self.scheduler = simulator.scheduler
        self.config = json.loads(Path(filename).read_text())
        ids = sorted(self.scheduler._replica_schedulers)
        if simulator._config.cluster_config.replica_config.num_pipeline_stages != 1:
            raise ValueError("Host churn initially supports pipeline size 1 only")
        scheduler_name = type(
            next(iter(self.scheduler._replica_schedulers.values()))
        ).__name__
        if scheduler_name not in {"VLLMReplicaScheduler", "SarathiReplicaScheduler"}:
            raise ValueError("Host churn supports vLLM and Sarathi replica schedulers")
        if type(self.scheduler).__name__ != "RoundRobinGlobalScheduler":
            raise ValueError("Host churn initially supports round-robin global routing")
        hosts = self.config.get("hosts")
        if hosts is None:
            count = self.config.get("workers_per_host", 1)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or len(ids) % count
            ):
                raise ValueError(
                    "workers_per_host must be a positive integer dividing num_replicas"
                )
            hosts = {
                f"h{i // count}": ids[i : i + count] for i in range(0, len(ids), count)
            }
        flat = [replica for replicas in hosts.values() for replica in replicas]
        if (
            sorted(flat) != ids
            or len(set(flat)) != len(flat)
            or any(not v for v in hosts.values())
        ):
            raise ValueError("hosts must partition all replica IDs exactly once")
        self.hosts = hosts
        self.mode = self.config.get("recovery", "restart")
        if self.mode not in {"restart", "ideal"}:
            raise ValueError("recovery must be restart or ideal")
        self.delay = seconds(self.config.get("detection_delay_s", 0)) + seconds(
            self.config.get("retry_delay_s", 0)
        )
        offline = self.config.get("initially_offline", [])
        if not set(offline) <= hosts.keys():
            raise ValueError("Unknown initially_offline host")
        self.ready = {
            r for h, replicas in hosts.items() if h not in offline for r in replicas
        }
        self.generation = dict.fromkeys(ids, 0)
        self.host_generation = dict.fromkeys(hosts, 0)
        self.records, self.requests, self.lifecycle = {}, {}, []
        self.cancelled_events = 0
        # Routing policy lives only in this optional extension.
        self.scheduler._host_churn = self
        self.scheduler.schedule = MethodType(
            lambda scheduler: self.route(), self.scheduler
        )
        self.events = []
        for item in self.config.get("events", []):
            if item["host"] not in hosts or item["action"] not in {"remove", "start"}:
                raise ValueError(
                    "Each event needs a known host and action remove/start"
                )
            time = seconds(item["time_s"])
            startup = seconds(item.get("startup_delay_s", 0))
            self.events.append(
                ChurnEvent(time, item["action"], item["host"], generation=startup)
            )

    def initial_events(self):
        return self.events

    def stamp(self, event):
        replica = getattr(event, "_replica_id", None)
        if replica is not None:
            event._host_generation = self.generation[replica]

    def valid(self, event):
        replica = getattr(event, "_replica_id", None)
        valid = replica is None or (
            replica in self.ready and event._host_generation == self.generation[replica]
        )
        if not valid:
            self.cancelled_events += 1
        return valid

    def route(self):
        eligible = sorted(self.ready)
        if not eligible:
            return []
        self.scheduler.sort_requests()
        mapping = []
        while self.scheduler._request_queue:
            request = self.scheduler._request_queue.pop(0)
            replica = eligible[self.scheduler._request_counter % len(eligible)]
            self.scheduler._request_counter += 1
            mapping.append((replica, request))
            self.requests[request.id] = request
            previous = self.records.setdefault(request.id, [])
            retry_wait = self.simulator._time - previous[-1]["end_s"] if previous else 0
            previous.append(
                {
                    "replica": replica,
                    "dispatch_s": self.simulator._time,
                    "execution_start_s": request._execution_time,
                    "retry_wait_s": retry_wait,
                    "execution_s": 0,
                    "queue_s": 0,
                    "status": "pending",
                }
            )
        return mapping

    def after_event(self, event):
        if type(event).__name__ == "RequestArrivalEvent":
            self.requests[event._request.id] = event._request
            self.records.setdefault(event._request.id, [])
        # First iteration start measures worker queueing for this attempt.
        if type(event).__name__ == "ReplicaStageScheduleEvent" and event._batch_stage:
            for request in event._batch_stage.requests:
                record = self.records[request.id][-1]
                if "first_execution_s" not in record:
                    record["first_execution_s"] = event.time
                    record["queue_s"] = event.time - record["dispatch_s"]
        if type(event).__name__ == "BatchEndEvent":
            for request in event._batch.requests:
                record = self.records[request.id][-1]
                record["execution_s"] = (
                    request._execution_time - record["execution_start_s"]
                )
                if request.completed:
                    record.update(status="completed", end_s=event.time)

    def handle(self, event):
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        if event.action == "retry":
            self.scheduler.add_request(event.request)
            return [GlobalScheduleEvent(event.time)]
        replicas = self.hosts[event.host]
        if event.action == "ready":
            if event.generation != self.host_generation[event.host]:
                return []
            self.ready.update(replicas)
            self.lifecycle.append(event.to_dict())
            return [GlobalScheduleEvent(event.time)]
        self.host_generation[event.host] += 1
        self.lifecycle.append(event.to_dict())
        retries = []
        for replica in replicas:
            self.ready.discard(replica)
            self.generation[replica] += 1
            worker = self.scheduler.get_replica_scheduler(replica)
            requests = {
                r.id: r for r in worker._request_queue + worker._preempted_requests
            }
            for batch in worker._active_batches.values():
                requests.update({r.id: r for r in batch.requests if not r.completed})
            for stage in worker._replica_stage_schedulers.values():
                active = stage._active_stage
                if active is not None:
                    elapsed = max(
                        0, min(event.time - active.scheduled_at, active.execution_time)
                    )
                    for request in active.requests:
                        request._execution_time += elapsed
                        # Incomplete model time is not inferred proportionally.
                stage._batch_queue.clear()
                stage._is_busy = False
                stage._active_stage = None
            worker._request_queue.clear()
            worker._preempted_requests.clear()
            worker._active_batches.clear()
            worker._allocation_map.clear()
            worker._num_allocated_blocks = 0
            worker._num_running_batches = 0
            for request in requests.values():
                record = self.records[request.id][-1]
                record.update(
                    status="interrupted",
                    end_s=event.time,
                    execution_s=request._execution_time - record["execution_start_s"],
                    processed_tokens=request.num_processed_tokens,
                )
                if "first_execution_s" not in record:
                    record["queue_s"] = event.time - record["dispatch_s"]
                request.retry_after_host_loss(self.mode == "ideal")
                retries.append(
                    ChurnEvent(event.time + self.delay, "retry", request=request)
                )
        if event.action == "start":
            retries.append(
                ChurnEvent(
                    event.time + event.generation,
                    "ready",
                    event.host,
                    generation=self.host_generation[event.host],
                )
            )
        return retries + [GlobalScheduleEvent(event.time)]

    def write_output(self):
        output = Path(self.simulator._config.metrics_config.output_dir)
        with (output / "churn_requests.jsonl").open("w") as stream:
            for request_id, request in sorted(self.requests.items()):
                row = {
                    "id": request_id,
                    "arrived_at": request.arrived_at,
                    "completed": request.completed,
                    "latency_s": (
                        request._completed_at - request.arrived_at
                        if request.completed
                        else None
                    ),
                    "recovery": self.mode,
                    "prefill_tokens": request._original_prefill_tokens,
                    "decode_tokens": request._original_decode_tokens,
                    "attempts": self.records[request_id],
                }
                stream.write(json.dumps(row) + "\n")
        (output / "churn_summary.json").write_text(
            json.dumps(
                {
                    "config": self.config,
                    "hosts": self.hosts,
                    "events": self.lifecycle,
                    "cancelled_events": self.cancelled_events,
                    "completed": sum(r.completed for r in self.requests.values()),
                    "interrupted_attempts": sum(
                        a["status"] == "interrupted"
                        for attempts in self.records.values()
                        for a in attempts
                    ),
                },
                indent=2,
            )
        )
