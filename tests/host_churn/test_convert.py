import csv
import json

import pytest

from vidur.host_churn.convert_requests import convert


def write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


def request(**changes):
    return dict(
        request_id="r1",
        phase="replay",
        scheduled_offset_seconds=2,
        trace_index=1,
        input_tokens=10,
        requested_output_tokens=128,
        actual_output_tokens=12,
        **changes
    )


def test_keeps_failed_requests_and_scheduled_arrivals(tmp_path):
    source, output = tmp_path / "requests.jsonl", tmp_path / "trace.csv"
    a = request()
    b = {
        **a,
        "request_id": "r0",
        "trace_index": 0,
        "actual_input_tokens": 20,
        "status_code": 500,
        "started_at_unix": 999,
        "attempts": 3,
    }
    write(source, [a, b, {**a, "phase": "warmup"}])
    assert convert(source, output) == (2, 148)
    rows = list(csv.DictReader(output.open()))
    assert [row["request_id"] for row in rows] == ["r0", "r1"]
    assert rows[0]["arrived_at"] == "2.0"
    assert rows[0]["num_decode_tokens"] == "128"


def test_actual_decode_opt_in(tmp_path):
    source, output = tmp_path / "r.jsonl", tmp_path / "t.csv"
    write(source, [request()])
    assert convert(source, output, "actual") == (1, 22)


def test_duplicate_attempts_rejected(tmp_path):
    source, output = tmp_path / "r.jsonl", tmp_path / "t.csv"
    write(source, [request(), request()])
    with pytest.raises(ValueError, match="Duplicate"):
        convert(source, output)
    assert not output.exists()


def test_missing_schedule_rejected(tmp_path):
    source, output = tmp_path / "r.jsonl", tmp_path / "t.csv"
    row = request()
    del row["scheduled_offset_seconds"]
    write(source, [row])
    with pytest.raises(ValueError, match="scheduled_offset_seconds"):
        convert(source, output)
