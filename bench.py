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
RLE_ALGORITHMS = ("rle",)
MODES = ("sequential", "parallel")
LIFECYCLES = ("reuse", "fresh")
STAGES = ("threshold_ms", "ccl_ms", "histogram_ms", "color_selection_ms", "final_coloring_ms")
STAGE_LABELS = ("Threshold", "CCL", "Histogram", "Selection", "Coloring")
RLE_STAGES = ("setup_ms", "scan_ccl_histogram_ms", "root_resolution_ms",
              "color_selection_ms", "final_coloring_ms")
RLE_STAGE_LABELS = ("Setup", "Scan/CCL/paint", "Reconciliation", "Selection", "Coloring")


def timing_stages(metadata):
    return RLE_STAGES if metadata.get("algorithm_requested") in RLE_ALGORITHMS else STAGES


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


def validate_measurements(data, algorithm, mode, runs, warmup, lifecycle="reuse"):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported measurements schema; rebuild colorist")
    metadata = data["metadata"]
    samples = data["samples"]
    if (metadata["algorithm_requested"] != algorithm or
            bool(metadata["parallel_requested"]) != (mode == "parallel") or
            metadata["measured_runs"] != runs or metadata["warmup_runs"] != warmup or
            len(samples) != runs):
        raise ValueError("Measurements do not match the requested experiment")
    if metadata.get("lifecycle", "reuse") != lifecycle:
        raise ValueError("Measurements do not match the requested lifecycle")
    expected = "base_sauf" if algorithm == "base_bbdt" and metadata["connectivity"] == 4 else algorithm
    if metadata["algorithm_effective"] != expected:
        raise ValueError("Unexpected effective labeling algorithm")
    if metadata["debug_enabled"]:
        raise ValueError("Benchmark requires debugging disabled: rebuild without -AlgoDebug / --algo-debug")
    if metadata["build_configuration"] not in ("Release", "RelWithDebInfo", "MinSizeRel"):
        raise ValueError("Benchmark requires an optimized build (e.g. Release)")
    stages = timing_stages(metadata)
    if metadata["timing_enabled"] and tuple(metadata.get("stage_keys", stages)) != stages:
        raise ValueError("Unexpected timing stage keys")
    keys = ("total_ms",) + (stages if metadata["timing_enabled"] else ())
    for sample in samples:
        sample_keys = keys + (("teardown_ms",) if lifecycle == "fresh" or "teardown_ms" in sample else ())
        for key in sample_keys:
            value = sample[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid timing: {key}={value}")
        if sample.get("teardown_ms", 0) > sample["total_ms"]:
            raise ValueError("Instance cleanup cannot exceed total algorithm time")
        if lifecycle == "reuse" and sample.get("teardown_ms", 0) != 0:
            raise ValueError("Reuse samples must not time instance destruction")
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
    metadata = experiment["metadata"]
    if requested == "parallel" and metadata.get("parallel_coloring") and not metadata["parallel_labeling"]:
        return "parallel (coloring)"
    actual = "parallel" if metadata["parallel_labeling"] or metadata.get("parallel_coloring") else "sequential"
    return requested if requested == actual else f"{requested} -> {actual}"


def experiment_name(algorithm, mode, lifecycle):
    return f"{algorithm}-{mode}" + ("-fresh" if lifecycle == "fresh" else "")


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
        for position, configuration in enumerate(order, 1):
            algorithm, mode = configuration[:2]
            lifecycle = configuration[2] if len(configuration) > 2 else "reuse"
            schedule.append({"round": round_index + 1, "position": position,
                             "algorithm": algorithm, "mode": mode, "lifecycle": lifecycle,
                             "measured_runs": per_round + int(round_index < extra)})
    return schedule


def merge_batch(experiment, metadata, samples, job, comparison, result):
    if experiment is None:
        experiment = {"mode": job["mode"], "lifecycle": job.get("lifecycle", "reuse"),
                      "metadata": dict(metadata), "samples": [],
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
    experiment["teardown_statistics"] = summarize([sample.get("teardown_ms", 0) for sample in experiment["samples"]])
    experiment["stage_statistics"] = {
        stage: summarize([sample[stage] for sample in experiment["samples"]])
        for stage in timing_stages(metadata) if metadata["timing_enabled"]
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
    all_stages = tuple(dict.fromkeys(stage for experiment in experiments
                                    for stage in timing_stages(experiment["metadata"])))
    for experiment in experiments:
        metadata = experiment["metadata"]
        identity = {"algorithm": metadata["algorithm_requested"],
                    "effective_algorithm": metadata["algorithm_effective"],
                    "mode": experiment["mode"],
                    "lifecycle": experiment["lifecycle"],
                    "labeling": "parallel" if metadata["parallel_labeling"] else "sequential",
                    "coloring": "parallel" if metadata.get("parallel_coloring") else "sequential",
                    "threads": metadata["opencv_threads"], "rounds": len(experiment["rounds"])}
        comparison = experiment["comparison"]
        summaries.append({**identity, "runs": len(experiment["samples"]),
                          **experiment["statistics"],
                          "teardown_median_ms": experiment["teardown_statistics"]["median_ms"],
                          "reference": comparison_label(comparison),
                          "differing_pixels": comparison["differing_pixels"] if comparison else "",
                          "difference_percent": comparison["difference_percent"] if comparison else ""})
        for iteration, sample in enumerate(experiment["samples"], 1):
            raw.append({**identity, "iteration": iteration, "round": sample["round"],
                        "position": sample["position"], "iteration_in_round": sample["iteration_in_round"],
                        "total_ms": sample["total_ms"],
                        "teardown_ms": sample.get("teardown_ms", 0),
                        **{stage: sample.get(stage, "") for stage in all_stages}})
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
    parser.add_argument("--algos", nargs="+", choices=ALGORITHMS + RLE_ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--threads", type=int, help="OpenCV thread count for parallel experiments (default: automatic)")
    parser.add_argument("--lifecycles", nargs="+", choices=LIFECYCLES, default=list(LIFECYCLES),
                        help="Instance reuse/fresh modes (default: both)")
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
    if args.threads is not None and not 1 <= args.threads <= 2147483647:
        parser.error("--threads must be an integer between 1 and 2147483647")
    if args.threads is not None and "parallel" not in args.modes:
        parser.error("--threads requires a parallel mode")
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
    args.lifecycles = list(dict.fromkeys(args.lifecycles))
    protected = {args.exe, args.mask, args.color_mask, args.reference}
    destinations = [args.output_dir / name for name in ("summary.csv", "measurements.csv", "stages.csv", "report.json")]
    for algorithm, mode, lifecycle in itertools.product(args.algos, args.modes, args.lifecycles):
        name = experiment_name(algorithm, mode, lifecycle)
        destinations.extend(args.output_dir / f"{name}.{extension}" for extension in ("png", "json"))
    if any(path.resolve() in protected for path in destinations):
        parser.error("Output directory would overwrite an input, executable or reference; choose another directory")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_rgba(args.reference) if args.reference else None
    configurations = list(itertools.product(args.algos, args.modes, args.lifecycles))
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
        "lifecycles": args.lifecycles,
        "parallel_threads_requested": args.threads,
        "rounds": round_count, "requested_rounds": args.rounds,
        "shuffle_enabled": shuffle, "shuffle_seed": seed, "execution_order": schedule,
    }
    order_label = "Shuffled" if shuffle else "Ordered"
    seed_label = f" -- seed: {seed}" if shuffle else ""
    print(f"{order_label} rounds: {round_count} -- total runs/configuration: {args.runs}{seed_label}", flush=True)
    collected = {}
    for job in schedule:
        algorithm, mode, lifecycle = job["algorithm"], job["mode"], job["lifecycle"]
        configuration = (algorithm, mode, lifecycle)
        name = experiment_name(*configuration)
        result_path = args.output_dir / f"{name}.png"
        metrics_path = args.output_dir / f"{name}.json"
        command = [str(args.exe), "--mask", str(args.mask), "--color_mask", str(args.color_mask),
                   "--output", str(result_path), "--algo-type", algorithm, "--lifecycle", lifecycle,
                   "--warmup", str(args.warmup), "--runs", str(job["measured_runs"]), "--metrics", str(metrics_path)]
        if mode == "parallel":
            command.append("--parallel")
            if args.threads is not None:
                command.extend(["--threads", str(args.threads)])
        print(f"Round {job['round']}/{round_count}, {job['position']}/{len(configurations)}: "
              f"{algorithm} / {mode} / {lifecycle} ({args.warmup} warm-up + {job['measured_runs']} measured)...", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode:
            raise RuntimeError(f"{name} exited with {completed.returncode}:\n{completed.stderr or completed.stdout}")
        with metrics_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        metadata, samples = validate_measurements(data, algorithm, mode, job["measured_runs"], args.warmup,
                                                 lifecycle)
        comparison = compare_images(load_rgba(result_path), reference) if reference is not None else None
        experiment = merge_batch(collected.get(configuration), metadata, samples, job, comparison, result_path)
        collected[configuration] = experiment
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
                     experiment["lifecycle"],
                     meta["opencv_threads"], len(experiment["samples"]),
                     f"{stats['median_ms']:.3f}", f"{stats['mean_ms']:.3f}", f"{stats['stddev_ms']:.3f}",
                     f"{stats['p95_ms']:.3f}", f"{stats['min_ms']:.3f}", f"{stats['max_ms']:.3f}",
                     f"{experiment['teardown_statistics']['median_ms']:.3f}",
                     comparison_label(experiment["comparison"])])
    print_table(["Algorithm", "Effective", "Mode", "Lifecycle", "Threads", "Runs", "Median", "Mean", "Stddev", "P95", "Min", "Max", "Cleanup", "Reference"], rows)
    for stages, labels, title in ((STAGES, STAGE_LABELS, "Median stage times (ms):"),
                                  (RLE_STAGES, RLE_STAGE_LABELS, "Median RLE stage times (ms):")):
        stage_rows = [[experiment["metadata"]["algorithm_requested"].removeprefix("base_"), execution_label(experiment),
                       experiment["lifecycle"],
                       *(f"{experiment['stage_statistics'][stage]['median_ms']:.3f}" for stage in stages)]
                      for experiment in experiments if experiment["stage_statistics"]
                      and timing_stages(experiment["metadata"]) == stages]
        if stage_rows:
            print(f"\n{title}")
            print_table(["Algorithm", "Mode", "Lifecycle", *labels], stage_rows)
    if not any(experiment["stage_statistics"] for experiment in experiments):
        print("\nStage timings unavailable: rebuild with -AlgoTiming / --algo-timing.")
    print("\nFresh totals include instance cleanup; Cleanup shows its median. Stage timings cover Compute only.")
    print(f"\nResults, raw JSON, CSV statistics and report.json: {args.output_dir}")
    return 1 if any(experiment["comparison"] and not experiment["comparison"]["match"]
                    for experiment in experiments) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
