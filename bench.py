#!/usr/bin/env python3
"""Run shuffled rounds of colorist experiments; timings come from C++."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys


ALGORITHMS = ("base_spaghetti", "base_sauf", "base_bbdt")
MODES = ("sequential", "parallel")
STAGES = ("threshold_ms", "ccl_ms", "histogram_ms", "color_selection_ms", "final_coloring_ms")
STAGE_LABELS = ("Threshold", "CCL", "Histogram", "Selection", "Coloring")


def summarize(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "stddev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],  # Nearest-rank percentile.
        "min_ms": min(values),
        "max_ms": max(values),
    }


def load_rgba(path):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Reference comparison requires Pillow: install requirements-bench.txt") from error
    with Image.open(path) as image:
        return image.convert("RGBA")


def compare_images(actual, reference):
    """Count differing pixels across all four channels, including hidden RGB."""
    from PIL import ImageChops

    if actual.size != reference.size:
        return {"match": False, "reason": "size mismatch", "differing_pixels": None,
                "difference_percent": None, "result_size": actual.size,
                "reference_size": reference.size}
    difference = ImageChops.difference(actual, reference)
    differing = difference.getchannel("R")
    for channel in ("G", "B", "A"):
        differing = ImageChops.lighter(differing, difference.getchannel(channel))
    pixels = actual.width * actual.height
    count = pixels - differing.histogram()[0]
    return {"match": count == 0, "reason": "equal" if count == 0 else "pixel mismatch",
            "differing_pixels": count, "difference_percent": 100.0 * count / pixels}


def validate_measurements(data, algorithm, mode, runs, warmup):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported measurements schema; rebuild colorist")
    metadata = data["metadata"]
    samples = data["samples"]
    if (metadata["algorithm_requested"] != algorithm or
            bool(metadata["parallel_requested"]) != (mode == "parallel") or
            metadata["measured_runs"] != runs or metadata["warmup_runs"] != warmup or
            len(samples) != runs):
        raise ValueError("Measurements do not match the requested experiment")
    expected = "base_sauf" if algorithm == "base_bbdt" and metadata["connectivity"] == 4 else algorithm
    if metadata["algorithm_effective"] != expected:
        raise ValueError("Unexpected effective labeling algorithm")
    if metadata["debug_enabled"]:
        raise ValueError("Benchmark requires debugging disabled: rebuild without -AlgoDebug / --algo-debug")
    if metadata["build_configuration"] not in ("Release", "RelWithDebInfo", "MinSizeRel"):
        raise ValueError("Benchmark requires an optimized build (e.g. Release)")
    keys = ("total_ms",) + (STAGES if metadata["timing_enabled"] else ())
    for sample in samples:
        for key in keys:
            value = sample[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid timing: {key}={value}")
    return metadata, samples


def print_table(headers, rows):
    strings = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in strings)) for i, header in enumerate(headers)]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in strings:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def comparison_label(comparison):
    if comparison is None:
        return "skipped"
    if comparison["match"]:
        return "PASS"
    if comparison["differing_pixels"] is None:
        return "FAIL (size)"
    return f"FAIL ({comparison['differing_pixels']} px, {comparison['difference_percent']:.3f}%)"


def execution_label(experiment):
    requested = experiment["mode"]
    actual = "parallel" if experiment["metadata"]["parallel_labeling"] else "sequential"
    return requested if requested == actual else f"{requested} -> {actual}"


def build_schedule(configurations, runs, rounds, seed, shuffle=True):
    if runs < 1 or rounds < 1:
        raise ValueError("Runs and rounds must be >= 1")
    round_count = min(rounds, runs)
    per_round, extra = divmod(runs, round_count)
    generator = random.Random(seed) if shuffle else None
    schedule = []
    for round_index in range(round_count):
        order = list(configurations)
        if generator is not None:
            generator.shuffle(order)
        for position, (algorithm, mode) in enumerate(order, 1):
            schedule.append({"round": round_index + 1, "position": position,
                             "algorithm": algorithm, "mode": mode,
                             "measured_runs": per_round + int(round_index < extra)})
    return schedule


def merge_batch(experiment, metadata, samples, job, comparison, result):
    if experiment is None:
        experiment = {"mode": job["mode"], "metadata": dict(metadata), "samples": [],
                      "rounds": [], "comparison": None, "result": str(result)}
        experiment["metadata"]["warmup_runs"] = 0
        experiment["metadata"]["measured_runs"] = 0
    for key, value in metadata.items():
        if key not in ("warmup_runs", "measured_runs") and experiment["metadata"][key] != value:
            raise ValueError(f"Configuration changed between rounds: {key}")
    experiment["metadata"]["warmup_runs"] += metadata["warmup_runs"]
    experiment["metadata"]["measured_runs"] += len(samples)
    experiment["samples"].extend({**sample, "round": job["round"], "position": job["position"],
                                  "iteration_in_round": iteration}
                                 for iteration, sample in enumerate(samples, 1))
    experiment["rounds"].append({"round": job["round"], "position": job["position"],
                                "warmup_runs": metadata["warmup_runs"], "measured_runs": len(samples),
                                "comparison": comparison})
    # Keep the first failure visible even when subsequent rounds match the reference.
    if experiment["comparison"] is None or experiment["comparison"]["match"]:
        experiment["comparison"] = comparison
    experiment["statistics"] = summarize([sample["total_ms"] for sample in experiment["samples"]])
    experiment["stage_statistics"] = {
        stage: summarize([sample[stage] for sample in experiment["samples"]])
        for stage in STAGES if metadata["timing_enabled"]
    }
    return experiment


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_reports(directory, context, experiments):
    summaries, raw, stages = [], [], []
    for experiment in experiments:
        metadata = experiment["metadata"]
        identity = {"algorithm": metadata["algorithm_requested"],
                    "effective_algorithm": metadata["algorithm_effective"],
                    "mode": experiment["mode"],
                    "labeling": "parallel" if metadata["parallel_labeling"] else "sequential",
                    "threads": metadata["opencv_threads"], "rounds": len(experiment["rounds"])}
        comparison = experiment["comparison"]
        summaries.append({**identity, "runs": len(experiment["samples"]),
                          **experiment["statistics"], "reference": comparison_label(comparison),
                          "differing_pixels": comparison["differing_pixels"] if comparison else "",
                          "difference_percent": comparison["difference_percent"] if comparison else ""})
        for iteration, sample in enumerate(experiment["samples"], 1):
            raw.append({**identity, "iteration": iteration, "round": sample["round"],
                        "position": sample["position"], "iteration_in_round": sample["iteration_in_round"],
                        "total_ms": sample["total_ms"],
                        **{stage: sample.get(stage, "") for stage in STAGES}})
        for stage, values in experiment["stage_statistics"].items():
            stages.append({**identity, "stage": stage, **values})
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "measurements.csv", raw)
    write_csv(directory / "stages.csv", stages)
    with (directory / "report.json").open("w", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, **context, "experiments": experiments}, stream,
                  indent=2, allow_nan=False)
        stream.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build/colorist.exe" if os.name == "nt" else "build/colorist"))
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--color-mask", type=Path, required=True)
    parser.add_argument("--algos", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up runs per configuration per round (default: 3)")
    parser.add_argument("--runs", type=int, default=30, help="Total measured runs per configuration (default: 30)")
    parser.add_argument("--rounds", type=int, default=10, help="Experiment rounds, capped at --runs (default: 10)")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep configured algorithm/mode order in every round")
    parser.add_argument("--seed", type=int, help="Shuffle seed; omitted generates and records a random seed")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/bench"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.rounds < 1:
        parser.error("--runs and --rounds must be >= 1 and --warmup must be >= 0")
    for name in ("exe", "mask", "color_mask", "reference"):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve()
            if not path.is_file():
                parser.error(f"File not found: {path}")
            setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    args.algos = list(dict.fromkeys(args.algos))
    args.modes = list(dict.fromkeys(args.modes))
    protected = {args.exe, args.mask, args.color_mask, args.reference}
    destinations = [args.output_dir / name for name in ("summary.csv", "measurements.csv", "stages.csv", "report.json")]
    for algorithm, mode in itertools.product(args.algos, args.modes):
        destinations.extend(args.output_dir / f"{algorithm}-{mode}.{extension}" for extension in ("png", "json"))
    if any(path.resolve() in protected for path in destinations):
        parser.error("Output directory would overwrite an input, executable or reference; choose another directory")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_rgba(args.reference) if args.reference else None
    configurations = list(itertools.product(args.algos, args.modes))
    shuffle = not args.no_shuffle
    seed = (args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)) if shuffle else None
    schedule = build_schedule(configurations, args.runs, args.rounds, seed, shuffle=shuffle)
    round_count = min(args.rounds, args.runs)
    context = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {"system": platform.system(), "release": platform.release(),
                    "architecture": platform.machine(), "processor": platform.processor(),
                    "logical_cpus": os.cpu_count(), "python": platform.python_version()},
        "executable": file_identity(args.exe), "mask": file_identity(args.mask),
        "color_mask": file_identity(args.color_mask),
        "reference": file_identity(args.reference) if args.reference else None,
        "warmup_runs": args.warmup, "measured_runs": args.runs,
        "rounds": round_count, "requested_rounds": args.rounds,
        "shuffle_enabled": shuffle, "shuffle_seed": seed, "execution_order": schedule,
    }
    order_label = "Shuffled" if shuffle else "Ordered"
    seed_label = f" -- seed: {seed}" if shuffle else ""
    print(f"{order_label} rounds: {round_count} -- total runs/configuration: {args.runs}{seed_label}", flush=True)
    collected = {}
    for job in schedule:
        algorithm, mode = job["algorithm"], job["mode"]
        name = f"{algorithm}-{mode}"
        result_path = args.output_dir / f"{name}.png"
        metrics_path = args.output_dir / f"{name}.json"
        command = [str(args.exe), "--mask", str(args.mask), "--color_mask", str(args.color_mask),
                   "--output", str(result_path), "--algo-type", algorithm,
                   "--warmup", str(args.warmup), "--runs", str(job["measured_runs"]), "--metrics", str(metrics_path)]
        if mode == "parallel":
            command.append("--parallel")
        print(f"Round {job['round']}/{round_count}, {job['position']}/{len(configurations)}: "
              f"{algorithm} / {mode} ({args.warmup} warm-up + {job['measured_runs']} measured)...", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode:
            raise RuntimeError(f"{name} exited with {completed.returncode}:\n{completed.stderr or completed.stdout}")
        with metrics_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        metadata, samples = validate_measurements(data, algorithm, mode, job["measured_runs"], args.warmup)
        comparison = compare_images(load_rgba(result_path), reference) if reference is not None else None
        experiment = merge_batch(collected.get((algorithm, mode)), metadata, samples, job, comparison, result_path)
        collected[algorithm, mode] = experiment
        # Retain all completed rounds in the configuration JSON, not just its last app invocation.
        with metrics_path.open("w", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, **experiment}, stream, indent=2, allow_nan=False)
            stream.write("\n")

    experiments = [collected[configuration] for configuration in configurations]
    save_reports(args.output_dir, context, experiments)
    print("\nTotal algorithm times (ms; sample standard deviation):")
    rows = []
    for experiment in experiments:
        meta, stats = experiment["metadata"], experiment["statistics"]
        rows.append([meta["algorithm_requested"].removeprefix("base_"),
                     meta["algorithm_effective"].removeprefix("base_"), execution_label(experiment),
                     meta["opencv_threads"], len(experiment["samples"]),
                     f"{stats['median_ms']:.3f}", f"{stats['mean_ms']:.3f}", f"{stats['stddev_ms']:.3f}",
                     f"{stats['p95_ms']:.3f}", f"{stats['min_ms']:.3f}", f"{stats['max_ms']:.3f}",
                     comparison_label(experiment["comparison"])])
    print_table(["Algorithm", "Effective", "Mode", "Threads", "Runs", "Median", "Mean", "Stddev", "P95", "Min", "Max", "Reference"], rows)
    stage_rows = [[experiment["metadata"]["algorithm_requested"].removeprefix("base_"), execution_label(experiment),
                   *(f"{experiment['stage_statistics'][stage]['median_ms']:.3f}" for stage in STAGES)]
                  for experiment in experiments if experiment["stage_statistics"]]
    if stage_rows:
        print("\nMedian stage times (ms):")
        print_table(["Algorithm", "Mode", *STAGE_LABELS], stage_rows)
    else:
        print("\nStage timings unavailable: rebuild with -AlgoTiming / --algo-timing.")
    print(f"\nResults, raw JSON, CSV statistics and report.json: {args.output_dir}")
    return 1 if any(experiment["comparison"] and not experiment["comparison"]["match"]
                    for experiment in experiments) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
