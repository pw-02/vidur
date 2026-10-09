"""Matched comparisons: python -m vidur.host_churn.compare R01=path R02=path."""

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    return {
        row["id"]: row
        for line in Path(path).read_text().splitlines()
        if (row := json.loads(line))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Short run ID=churn_requests.jsonl")
    parser.add_argument(
        "--after", type=float, default=0, help="Scheduled arrival cutoff in seconds"
    )
    parser.add_argument("--baseline", default="R02")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.runs)
    rows = {name: load(path) for name, path in runs.items()}
    if args.baseline not in rows:
        parser.error("baseline must name a supplied run")
    baseline = rows[args.baseline]
    ids = set(baseline)
    for run in rows.values():
        if set(run) != ids:
            parser.error("runs must contain the same request IDs")
        for key in ids:
            if (
                run[key]["arrived_at"],
                run[key]["prefill_tokens"],
                run[key]["decode_tokens"],
            ) != (
                baseline[key]["arrived_at"],
                baseline[key]["prefill_tokens"],
                baseline[key]["decode_tokens"],
            ):
                parser.error("request arrival schedules or token lengths do not match")
    selected = [key for key in ids if baseline[key]["arrived_at"] >= args.after]
    print("Run index")
    for name, path in runs.items():
        print(f"{name}: {path}")
    print(
        "\n| Run | Matched requests | Completed | Mean latency (s) | Interrupted attempts |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, run in rows.items():
        completed = [run[key]["latency_s"] for key in selected if run[key]["completed"]]
        mean = f"{statistics.mean(completed):.3f}" if completed else "n/a"
        interrupted = sum(
            a["status"] == "interrupted"
            for key in selected
            for a in run[key]["attempts"]
        )
        print(
            f"| {name} | {len(selected)} | {len(completed)} | {mean} | {interrupted} |"
        )
    print("\nMatched interrupted requests (all arrival times)")
    print(
        "| Run | Interrupted requests | Completed in both | Mean latency (s) | Baseline mean (s) |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, run in rows.items():
        if name == args.baseline:
            continue
        interrupted_ids = [
            key
            for key in ids
            if any(a["status"] == "interrupted" for a in run[key]["attempts"])
        ]
        matched = [
            key
            for key in interrupted_ids
            if run[key]["completed"] and baseline[key]["completed"]
        ]
        run_mean = (
            f"{statistics.mean(run[key]['latency_s'] for key in matched):.3f}"
            if matched
            else "n/a"
        )
        base_mean = (
            f"{statistics.mean(baseline[key]['latency_s'] for key in matched):.3f}"
            if matched
            else "n/a"
        )
        print(
            f"| {name} | {len(interrupted_ids)} | {len(matched)} | {run_mean} | {base_mean} |"
        )
    print(
        "\nLatency means include completed requests only; compare completion counts before interpreting them."
    )


if __name__ == "__main__":
    main()
