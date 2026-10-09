"""Convert MalleServe logical-request JSONL into a Vidur arrival trace."""

import argparse
import csv
import json
import math
from pathlib import Path


def positive_tokens(row, fields):
    for field in fields:
        value = row.get(field)
        if value is not None:
            number = float(value)
            if math.isfinite(number) and number > 0 and number.is_integer():
                return int(number)
    raise ValueError(f"Missing positive integer token count: {', '.join(fields)}")


def convert(source, destination, decode="requested"):
    rows, seen = [], set()
    for line_number, line in enumerate(Path(source).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("phase", "replay") != "replay":
                continue
            request_id = row["request_id"]
            if request_id in seen:
                raise ValueError(
                    "Duplicate logical request ID; supply the request-summary file, not attempt logs"
                )
            seen.add(request_id)
            arrival = float(row["scheduled_offset_seconds"])
            if not math.isfinite(arrival) or arrival < 0:
                raise ValueError(
                    "scheduled_offset_seconds must be finite and nonnegative"
                )
            prefill = positive_tokens(row, ["actual_input_tokens", "input_tokens"])
            fields = (
                ["requested_output_tokens"]
                if decode == "requested"
                else ["actual_output_tokens"]
            )
            output = positive_tokens(row, fields)
            rows.append(
                (
                    arrival,
                    int(row.get("trace_index", line_number)),
                    request_id,
                    prefill,
                    output,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("No replay requests found")
    rows.sort(key=lambda row: (row[0], row[1]))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["arrived_at", "num_prefill_tokens", "num_decode_tokens", "request_id"]
        )
        writer.writerows(
            (arrival, prefill, output, request_id)
            for arrival, _, request_id, prefill, output in rows
        )
    return len(rows), max(prefill + output for _, _, _, prefill, output in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--decode",
        choices=["requested", "actual"],
        default="requested",
        help="Original output target (default), or measured completion length",
    )
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    try:
        count, maximum = convert(args.source, args.destination, args.decode)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {count} logical requests to {args.destination}")
    print(f"Set --trace_request_generator_config_max_tokens to at least {maximum}")
    print(
        "request_id is retained for reference; Vidur assigns numeric IDs by CSV row order."
    )


if __name__ == "__main__":
    main()
