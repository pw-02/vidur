"""Compare host-removal experiments: python -m vidur.host_churn.compare --help."""

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


def recovery_label(mode):
    return {
        "restart": "Retry interrupted requests from the beginning",
        "ideal": "Resume interrupted requests from completed progress (ideal)",
    }.get(mode, "Recovery behaviour not recorded")


def comparison_time_label(metadata, cutoff):
    is_removal = any(
        e.get("action") == "remove" and event_time(e) == cutoff
        for m in metadata.values()
        for e in m.get("events", [])
    )
    return "host removal" if is_removal else "comparison time"


def report(runs, paths, metadata, baseline, cutoff):
    ids = list(runs[baseline])
    base = runs[baseline]
    union = {i for i in ids if any(interrupted(run[i]) for run in runs.values())}
    time_label = comparison_time_label(metadata, cutoff)
    text = ["# Host-removal experiment results", "", "## Run index", ""]
    index = []
    for name, path in paths.items():
        config = metadata[name].get("config", {})
        actions = [
            f'{e.get("host", "Host")} {dict(remove="removed", start="starts initialising", ready="ready to serve")[e["action"]]} at {event_time(e):g}s'
            for e in metadata[name].get("events", [])
            if e.get("action") in ("remove", "start", "ready")
        ]
        mode = config.get(
            "recovery", next(iter(runs[name].values())).get("recovery", "unknown")
        )
        index.append(
            [
                name,
                ("Baseline; " if name == baseline else "") + recovery_label(mode),
                "; ".join(actions)
                or (
                    "No host changes"
                    if "events" in metadata[name]
                    else "Host events not recorded"
                ),
                str(path),
            ]
        )
    text += [
        table(
            [
                "Run",
                "How interrupted requests are retried",
                "Host changes",
                "Results file",
            ],
            index,
        ),
        "",
        f"Baseline: **{baseline}**. Time used to divide requests: **{cutoff:g} s** ({time_label}). "
        f"All runs contain the same {len(ids):,} request IDs, arrival times and token lengths.",
        "",
        "The simulator has two recovery options. **Retry from the beginning** discards "
        "the interrupted request’s completed progress. **Resume completed progress (ideal)** "
        "assumes that progress is available on the next worker, without modelling the cost "
        "of saving or transferring it. Work from an unfinished iteration is still lost. "
        "Both options still wait for a retry and share the remaining serving capacity.",
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
                "Completed / total",
                "Mean latency (s)",
                "p95 latency (s)",
                "Mean worker wait (s)",
                "Interrupted requests",
                "Interrupted attempts",
            ],
            results,
        ),
        "",
        "Latency runs from the scheduled arrival to final completion, including retries. "
        "Means above use completed requests only. One request can be interrupted more than "
        "once; interrupted attempts counts each interruption separately. Worker wait adds "
        "up the time waiting on a worker before each attempt first executes. It excludes "
        "retry delay, waiting before dispatch to a worker, and waits between iterations.",
        "",
        f"## Before and after {time_label}",
        "",
        "Each comparison uses the same request IDs in that run and the baseline, "
        "including only requests completed in both runs. Requests are grouped by scheduled "
        "arrival time. Extra latency is the run mean minus the baseline mean; positive "
        "means slower. Latency ratio is run mean divided by baseline mean (2× means twice as long).",
        "",
    ]
    cohorts = [
        ("All requests", ids),
        (
            f"Arrived before {time_label}",
            [i for i in ids if base[i]["arrived_at"] < cutoff],
        ),
        (
            f"Arrived at or after {time_label}",
            [i for i in ids if base[i]["arrived_at"] >= cutoff],
        ),
        (
            f"Arrived at or after {time_label}; not interrupted in any run",
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
                "Request group",
                "Run",
                "Requests",
                "Completed in both runs",
                "Mean latency (s)",
                "Baseline latency (s)",
                "Extra latency (s)",
                "Latency ratio",
            ],
            rows,
        ),
        "",
        "The last group contains requests that were not interrupted in any run. A slowdown "
        "here shows that later requests are also affected. They can still wait behind retried "
        "requests, so this group does not isolate the effect of losing capacity.",
        "",
        "## Interrupted requests",
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
                "Completed in both runs",
                "Mean latency (s)",
                "Baseline latency (s)",
                "Extra latency (s)",
                "Latency ratio",
            ],
            rows,
        ),
        "",
        "For each run, this table compares its interrupted requests with those same "
        "requests in the baseline. A request can arrive before host removal and still be "
        "interrupted. Different runs may interrupt different requests, so these rows do "
        "not directly compare the two ways of retrying them.",
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
        text += [
            "## Retrying from the beginning versus resuming completed progress",
            "",
        ]
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
                    "Runs compared",
                    "Request group",
                    "Requests",
                    "Completed in both runs",
                    "Retry from beginning: mean (s)",
                    "Resume progress: mean (s)",
                    "Extra latency from restarting (s)",
                    "Latency ratio",
                ],
                rows,
            ),
            "",
            "The first run in each pair retries from the beginning; the second resumes "
            "completed progress. Both columns use the same requests. A positive difference "
            "means retrying from the beginning took longer. Interpret this as the effect of "
            "the recovery option only if host changes, hardware and scheduler settings are "
            "also the same. The difference includes any resulting changes in queueing; it "
            "does not separate the total slowdown exactly into capacity loss and retry cost.",
            "",
        ]
    text += [
        "## What these results can tell us",
        "",
        "The report checks that requests, arrival times and token lengths agree across runs. "
        "It does not check all simulator settings or whether timing predictions match the real "
        "system. Default profiles can test the simulator, but do not establish Polaris/Qwen "
        "performance. Comparing host removal with the baseline includes both lost capacity "
        "and interruption recovery. p95 is the latency below which 95% of completed requests "
        "fall, calculated with linear interpolation.",
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
        ax.axvline(
            cutoff,
            color="0.3",
            linestyle="--",
            label=f"{comparison_time_label(metadata, cutoff).capitalize()} at {cutoff:g}s",
        )
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
            label=f'{"Host removed" if action == "remove" else "Host ready to serve"} at {time:g}s',
        )
    ax.set(
        xlabel="Scheduled arrival time (s)",
        ylabel="Mean latency (s)",
        title=f"Mean latency for requests arriving in each {bin_seconds:g}s interval",
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
        "--after",
        type=float,
        help="Divide requests by scheduled arrival time; defaults to the first recorded host removal",
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
                "comparison time must be finite and nonnegative; interval length must be finite and positive"
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
                    "\n## Figures\n\n![Latency of the same requests in each run](latency_summary.png)\n\n"
                    "![Latency by scheduled arrival](latency_over_time.png)\n\n"
                    "Both figures use the same requests, completed in every run. Each point in the "
                    "time plot is the mean latency of requests scheduled to arrive in that interval; "
                    "it does not show the queue length at that moment. Run IDs are explained in the run index.\n"
                )
        (args.output_dir / "report.md").write_text(result)
        print(result)
        print(f'\nSaved: {args.output_dir / "report.md"}')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
