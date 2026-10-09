import json
import sys

import pytest

from vidur.host_churn.compare import main


def write_run(path, arrivals=(0, 2), tokens=8):
    rows = [
        dict(
            id=i,
            arrived_at=t,
            prefill_tokens=tokens,
            decode_tokens=4,
            completed=True,
            latency_s=3,
            attempts=[{"status": "completed"}],
        )
        for i, t in enumerate(arrivals)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_matched_arrival_cohort(tmp_path, monkeypatch, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(a)
    write_run(b)
    monkeypatch.setattr(
        sys, "argv", ["compare", f"R01={a}", f"R02={b}", "--after", "1"]
    )
    main()
    output = capsys.readouterr().out
    assert "Run index" in output
    assert "| R01 | 1 | 1 | 3.000 | 0 |" in output


@pytest.mark.parametrize("arrivals,tokens", [((0, 3), 8), ((0, 2), 9)])
def test_rejects_unmatched_workloads(tmp_path, monkeypatch, arrivals, tokens):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(a)
    write_run(b, arrivals, tokens)
    monkeypatch.setattr(sys, "argv", ["compare", f"R01={a}", f"R02={b}"])
    with pytest.raises(SystemExit):
        main()
