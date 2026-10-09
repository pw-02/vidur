"""Matched churn reports: python -m vidur.host_churn.compare --help."""

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path):
    rows = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f'{path}:{number}: duplicate request ID {row["id"]}')
        if row["completed"] and (
            row.get("latency_s") is None
            or not math.isfinite(row["latency_s"])
            or row["latency_s"] < 0
        ):
            raise ValueError(f"{path}:{number}: invalid completed latency")
        rows[row["id"]] = row
    if not rows:
        raise ValueError(f"{path}: no requests")
    return rows


def interrupted(row):
    return any(a["status"] == "interrupted" for a in row["attempts"])


def mean(values):
    return statistics.mean(values) if values else None


def percentile(values, fraction=0.95):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[math.ceil(position)] - values[lower]) * (
        position - lower
    )


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    def line(row):
        return "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"

    return "\n".join(
        [line(headers), line(["---"] * len(headers))] + [line(r) for r in rows]
    )


def paired(run, reference, ids):
    selected = [i for i in ids if run[i]["completed"] and reference[i]["completed"]]
    values = [run[i]["latency_s"] for i in selected]
    base = [reference[i]["latency_s"] for i in selected]
    a, b = mean(values), mean(base)
    return [
        len(selected),
        fmt(a),
        fmt(b),
        fmt(a - b if selected else None),
        "—" if b in (None, 0) else f"{a / b:.2f}×",
    ]


def summary(path):
    source = path.parent / "churn_summary.json"
    result = json.loads(source.read_text()) if source.exists() else {}
    for event in result.get("events", []):
        event.setdefault("action", event.get("event_type"))
    return result


def event_time(event):
    return event.get("time_s", event.get("time"))


def report(runs, paths, metadata, baseline, cutoff):
    ids = list(runs[baseline])
    base = runs[baseline]
    union = {i for i in ids if any(interrupted(run[i]) for run in runs.values())}
    text = ["# Host-churn comparison", "", "## Run index", ""]
    index = []
    for name, path in paths.items():
        config = metadata[name].get("config", {})
        actions = [
            f'{e["action"]} {e.get("host", "")} at {event_time(e):g}s'
            for e in metadata[name].get("events", [])
            if e.get("action") in ("remove", "start", "ready")
        ]
        mode = config.get(
            "recovery", next(iter(runs[name].values())).get("recovery", "unknown")
        )
        index.append(
            [
                name,
                "baseline" if name == baseline else mode,
                "; ".join(actions) or "not recorded",
                str(path),
            ]
        )
    text += [
        table(["Run", "Recovery / reference", "Host events", "Source"], index),
        "",
        f"Baseline: **{baseline}**. Arrival cutoff: **{cutoff:g} s**. "
        f"All runs contain the same {len(ids):,} request IDs, arrival times and token lengths.",
        "",
        "## Overall results",
        "",
    ]
    results = []
    for name, run in runs.items():
        complete = [r for r in run.values() if r["completed"]]
        values = [r["latency_s"] for r in complete]
        queue = [sum(a.get("queue_s", 0) for a in r["attempts"]) for r in complete]
        has_queue = all("queue_s" in a for r in complete for a in r["attempts"])
        results.append(
            [
                name,
                f"{len(complete)}/{len(ids)}",
                fmt(mean(values)),
                fmt(percentile(values)),
                fmt(mean(queue)) if has_queue else "—",
                sum(interrupted(r) for r in run.values()),
                sum(
                    a["status"] == "interrupted"
                    for r in run.values()
                    for a in r["attempts"]
                ),
            ]
        )
    text += [
        table(
            [
                "Run",
                "Completed",
                "Mean latency (s)",
                "p95 (s)",
                "Mean summed attempt queue (s)",
                "Interrupted requests",
                "Interrupted attempts",
            ],
            results,
        ),
        "",
        "Latency and queue means above use completed requests only. Queue time is the sum "
        "of recorded worker waits before the first execution of each attempt; it excludes "
        "retry delay, global waiting and waits between iterations.",
        "",
        "## Matched comparisons against baseline",
        "",
        "Each row below uses only requests completed in both that run and the baseline. "
        "Δ is run minus baseline; positive means slower. The paired count can differ by run.",
        "",
    ]
    cohorts = [
        ("All requests", ids),
        ("Before cutoff", [i for i in ids if base[i]["arrived_at"] < cutoff]),
        ("At / after cutoff", [i for i in ids if base[i]["arrived_at"] >= cutoff]),
        (
            "At / after cutoff, never interrupted in any run",
            [i for i in ids if base[i]["arrived_at"] >= cutoff and i not in union],
        ),
    ]
    rows = []
    for label, cohort in cohorts:
        for name, run in runs.items():
            if name != baseline:
                rows.append([label, name, len(cohort)] + paired(run, base, cohort))
    text += [
        table(
            [
                "Cohort",
                "Run",
                "Selected",
                "Completed in both",
                "Run mean (s)",
                "Baseline mean (s)",
                "Mean Δ (s)",
                "Ratio",
            ],
            rows,
        ),
        "",
        "The never-interrupted cohort shows whether slowdown reaches requests that were "
        "not directly interrupted. Their latency can still include congestion caused by retries; "
        "excluding them does not remove retry effects from the system.",
        "",
        "## Requests interrupted in each run",
        "",
    ]
    rows = []
    for name, run in runs.items():
        if name == baseline:
            continue
        cohort = [i for i in ids if interrupted(run[i])]
        rows.append([name, len(cohort)] + paired(run, base, cohort))
    text += [
        table(
            [
                "Run",
                "Interrupted requests",
                "Completed in both",
                "Run mean (s)",
                "Baseline mean (s)",
                "Mean Δ (s)",
                "Ratio",
            ],
            rows,
        ),
        "",
        "These cohorts include all arrival times: a request can arrive before removal and "
        "still be interrupted. Each run may interrupt different IDs, so these rows are not "
        "a direct restart-versus-ideal comparison.",
        "",
    ]
    restarts = [
        n
        for n in runs
        if n != baseline
        and metadata[n]
        .get("config", {})
        .get("recovery", next(iter(runs[n].values())).get("recovery"))
        == "restart"
    ]
    ideals = [
        n
        for n in runs
        if n != baseline
        and metadata[n]
        .get("config", {})
        .get("recovery", next(iter(runs[n].values())).get("recovery"))
        == "ideal"
    ]
    if restarts and ideals:
        text += ["## Restart versus ideal recovery", ""]
        rows = []
        for restart in restarts:
            for ideal in ideals:
                for label, cohort in [
                    ("All requests", ids),
                    (
                        "Interrupted in either run",
                        [
                            i
                            for i in ids
                            if interrupted(runs[restart][i])
                            or interrupted(runs[ideal][i])
                        ],
                    ),
                ]:
                    rows.append(
                        [f"{restart} vs {ideal}", label, len(cohort)]
                        + paired(runs[restart], runs[ideal], cohort)
                    )
        text += [
            table(
                [
                    "Pair",
                    "Cohort",
                    "Selected",
                    "Completed in both",
                    "Restart mean (s)",
                    "Ideal mean (s)",
                    "Restart − ideal (s)",
                    "Ratio",
                ],
                rows,
            ),
            "",
            "Interpret this as a recovery-mode comparison only when both runs use the same "
            "host schedule, workload, hardware and scheduler settings. Ideal recovery preserves "
            "completed-iteration progress, but still includes capacity loss, interrupted-iteration "
            "loss and retry waiting. Changes in queueing are part of the difference; this is "
            "not an exact causal decomposition.",
            "",
        ]
    text += [
        "## Interpretation limits",
        "",
        "This report validates workload matching, not calibration or all simulator settings. "
        "Runs using default model and hardware profiles are simulator checks, not measured "
        "Polaris/Qwen results. A baseline-versus-removal difference combines capacity loss "
        "and recovery effects. p95 uses linear interpolation.",
        "",
    ]
    return "\n".join(text)


def plots(runs, metadata, baseline, cutoff, bin_seconds, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    names = list(runs)
    colors = ["#4c78a8", "#e45756", "#54a24b", "#b279a2", "#f2a541"]
    shared = [
        i for i in runs[baseline] if all(run[i]["completed"] for run in runs.values())
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, reducer, title in zip(
        axes, [mean, percentile], ["Mean latency", "p95 latency"]
    ):
        vals = [reducer([runs[n][i]["latency_s"] for i in shared]) for n in names]
        ax.bar(
            names,
            [v or 0 for v in vals],
            color=[colors[j % len(colors)] for j in range(len(names))],
        )
        for j, val in enumerate(vals):
            ax.annotate(
                fmt(val),
                (j, val or 0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
        ax.set(ylabel="Latency (s)", title=title)
        ax.margins(y=0.2)
    fig.suptitle(f"Same {len(shared):,} requests completed in every run")
    fig.tight_layout()
    fig.savefig(output / "latency_summary.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = {}
    for i in shared:
        bucket = math.floor(runs[baseline][i]["arrived_at"] / bin_seconds)
        bins.setdefault(bucket, []).append(i)
    for j, name in enumerate(names):
        buckets = sorted(bins)
        ax.plot(
            [(b + 0.5) * bin_seconds for b in buckets],
            [mean([runs[name][i]["latency_s"] for i in bins[b]]) for b in buckets],
            label=name,
            color=colors[j % len(colors)],
        )
    if cutoff > 0:
        ax.axvline(cutoff, color="0.3", linestyle="--", label=f"Cutoff {cutoff:g}s")
    # Mark each distinct observed removal / ready event once.
    events = {
        (e["action"], event_time(e))
        for m in metadata.values()
        for e in m.get("events", [])
        if e.get("action") in ("remove", "ready")
    }
    for action, time in sorted(events, key=lambda x: x[1]):
        if action == "remove" and time == cutoff:
            continue
        ax.axvline(
            time,
            color="0.6",
            linestyle=":" if action == "ready" else "--",
            label=f"{action} {time:g}s",
        )
    ax.set(
        xlabel="Scheduled arrival time (s)",
        ylabel="Mean latency (s)",
        title=f"Matched arrival bins ({bin_seconds:g}s); completed in every run",
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output / "latency_over_time.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positional_runs", nargs="*", metavar="RUN=PATH")
    parser.add_argument(
        "--runs", nargs="+", help="Short run ID=run folder or JSONL file"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("simulator_output"),
        help="Find latest baseline/removal/ideal subfolders when runs are omitted",
    )
    parser.add_argument(
        "--baseline", help="Baseline run ID (default: first supplied run)"
    )
    parser.add_argument(
        "--after", type=float, help="Arrival cutoff; default: first recorded removal"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("churn_comparison"))
    parser.add_argument("--bin-seconds", type=float, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.runs and args.positional_runs:
        parser.error("use positional runs or --runs, not both")
    supplied = args.runs or args.positional_runs
    try:
        if not supplied:
            supplied = []
            for name, folder in [
                ("R01", "baseline"),
                ("R02", "removal"),
                ("R03", "ideal"),
            ]:
                files = sorted((args.root / folder).glob("*/churn_requests.jsonl"))
                if not files:
                    raise ValueError(
                        f"no runs in {args.root / folder}; supply --runs explicitly"
                    )
                supplied.append(f"{name}={files[-1]}")
        paths = {}
        for item in supplied:
            name, value = item.split("=", 1)
            if not name or name in paths:
                raise ValueError(f"empty or duplicate run ID: {name}")
            path = Path(value)
            paths[name] = path / "churn_requests.jsonl" if path.is_dir() else path
        baseline = args.baseline or next(iter(paths))
        if baseline not in paths:
            raise ValueError("baseline must name a supplied run")
        runs = {n: load(p) for n, p in paths.items()}
        base = runs[baseline]
        for name, run in runs.items():
            if set(run) != set(base):
                raise ValueError(f"{name}: request IDs do not match baseline")
            for key in base:
                if tuple(
                    run[key][f]
                    for f in ("arrived_at", "prefill_tokens", "decode_tokens")
                ) != tuple(
                    base[key][f]
                    for f in ("arrived_at", "prefill_tokens", "decode_tokens")
                ):
                    raise ValueError(
                        f"{name}: arrival schedule or token lengths do not match for {key}"
                    )
        metadata = {n: summary(p) for n, p in paths.items()}
        removals = [
            event_time(e)
            for n, m in metadata.items()
            if n != baseline
            for e in m.get("events", [])
            if e.get("action") == "remove"
        ]
        cutoff = args.after if args.after is not None else min(removals, default=0)
        if (
            not math.isfinite(cutoff)
            or cutoff < 0
            or not math.isfinite(args.bin_seconds)
            or args.bin_seconds <= 0
        ):
            raise ValueError(
                "cutoff must be finite and nonnegative; bin size must be finite and positive"
            )
        result = report(runs, paths, metadata, baseline, cutoff)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_plots:
            try:
                plots(
                    runs, metadata, baseline, cutoff, args.bin_seconds, args.output_dir
                )
            except ImportError:
                print("Figures skipped: install matplotlib, or use --no-plots.")
            else:
                result += (
                    "\n## Figures\n\n![Matched latency summary](latency_summary.png)\n\n"
                    "![Latency by scheduled arrival](latency_over_time.png)\n\n"
                    "Figures use requests completed in every run. Arrival bins contain the same IDs "
                    "for all curves; they are not instantaneous queue measurements.\n"
                )
        (args.output_dir / "report.md").write_text(result)
        print(result)
        print(f'\nSaved: {args.output_dir / "report.md"}')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
