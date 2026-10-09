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
        sys,
        "argv",
        [
            "compare",
            f"R01={a}",
            f"R02={b}",
            "--after",
            "1",
            "--no-plots",
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "Run index" in output
    assert (
        "| At / after cutoff | R02 | 1 | 1 | 3.000 | 3.000 | 0.000 | 1.00× |" in output
    )


@pytest.mark.parametrize("arrivals,tokens", [((0, 3), 8), ((0, 2), 9)])
def test_rejects_unmatched_workloads(tmp_path, monkeypatch, arrivals, tokens):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(a)
    write_run(b, arrivals, tokens)
    monkeypatch.setattr(sys, "argv", ["compare", f"R01={a}", f"R02={b}"])
    with pytest.raises(SystemExit):
        main()


def test_report_pairing_and_discovery(tmp_path, monkeypatch, capsys):
    root = tmp_path / "runs"
    for folder, mode, values in [
        ("baseline", "restart", [1, 2, 3, 4]),
        ("removal", "restart", [5, 8, 9, None]),
        ("ideal", "ideal", [3, 4, 5, 6]),
    ]:
        path = root / folder / "2026-10-09_12-00-00"
        path.mkdir(parents=True)
        rows = []
        for i, latency in enumerate(values):
            rows.append(
                dict(
                    id=i,
                    arrived_at=i,
                    prefill_tokens=8,
                    decode_tokens=4,
                    completed=latency is not None,
                    latency_s=latency,
                    recovery=mode,
                    attempts=[
                        dict(
                            status=(
                                "interrupted"
                                if folder != "baseline" and i == 0
                                else "completed"
                            ),
                            queue_s=0.2,
                        )
                    ],
                )
            )
        (path / "churn_requests.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
        (path / "churn_summary.json").write_text(
            json.dumps(
                dict(
                    config=dict(recovery=mode),
                    events=(
                        []
                        if folder == "baseline"
                        else [dict(event_type="remove", host="h0", time=1)]
                    ),
                )
            )
        )
    out = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare", "--root", str(root), "--output-dir", str(out), "--no-plots"],
    )
    main()
    report = (out / "report.md").read_text()
    assert "Arrival cutoff: **1 s**" in report
    assert (
        "| At / after cutoff | R02 | 3 | 2 | 8.500 | 2.500 | 6.000 | 3.40× |" in report
    )
    assert (
        "| R02 vs R03 | Interrupted in either run | 1 | 1 | 5.000 | 3.000 | 2.000 | 1.67× |"
        in report
    )
    assert "| R02 | 1 | 1 | 5.000 | 1.000 | 4.000 | 5.00× |" in report
    assert "not an exact causal decomposition" in report


def test_duplicate_request_rejected(tmp_path):
    from vidur.host_churn.compare import load

    path = tmp_path / "requests.jsonl"
    write_run(path, arrivals=(0,))
    path.write_text(path.read_text() + "\n" + path.read_text())
    with pytest.raises(ValueError, match="duplicate request ID"):
        load(path)


def test_plot_uses_shared_completed_ids(tmp_path):
    from vidur.host_churn.compare import plots

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(a)
    write_run(b)
    from vidur.host_churn.compare import load

    runs = {"R01": load(a), "R02": load(b)}
    runs["R02"][1].update(completed=False, latency_s=None)
    plots(runs, {"R01": {}, "R02": {}}, "R01", 1, 10, tmp_path)
    assert (tmp_path / "latency_summary.png").stat().st_size > 1000
    assert (tmp_path / "latency_over_time.png").stat().st_size > 1000
