import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench

try:
    from PIL import Image
except ImportError:
    Image = None


class StatisticsTests(unittest.TestCase):
    def test_known_distribution(self):
        self.assertEqual(bench.summarize([1.0, 2.0, 3.0]), {
            "median_ms": 2.0, "mean_ms": 2.0, "stddev_ms": 1.0,
            "p95_ms": 3.0,
            "min_ms": 1.0, "max_ms": 3.0,
        })

    def test_single_measurement(self):
        self.assertEqual(bench.summarize([4.0])["stddev_ms"], 0.0)
        self.assertEqual(bench.summarize([4.0])["p95_ms"], 4.0)

    def test_p95_nearest_rank(self):
        self.assertEqual(bench.summarize(list(range(20, 0, -1)))["p95_ms"], 19)
        self.assertEqual(bench.summarize(list(range(1, 31)))["p95_ms"], 29)

    def test_sample_deviation(self):
        self.assertEqual(bench.summarize([1.0, 5.0])["stddev_ms"], statistics.stdev([1.0, 5.0]))


class ExecutionLabelTests(unittest.TestCase):
    def test_matching_modes_use_one_label(self):
        for mode, parallel in (("sequential", 0), ("parallel", 1)):
            self.assertEqual(bench.execution_label({"mode": mode, "metadata": {"parallel_labeling": parallel}}), mode)

    def test_parallel_fallback_is_visible(self):
        self.assertEqual(bench.execution_label({"mode": "parallel", "metadata": {"parallel_labeling": 0}}),
                         "parallel -> sequential")

    def test_parallel_coloring_is_visible_with_sequential_labeling(self):
        self.assertEqual(bench.execution_label({"mode": "parallel", "metadata": {
            "parallel_labeling": 0, "parallel_coloring": 1}}), "parallel (coloring)")


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.configurations = [(algorithm, mode) for algorithm in bench.ALGORITHMS for mode in bench.MODES]

    def test_each_round_covers_matrix_and_preserves_total_runs(self):
        schedule = bench.build_schedule(self.configurations, runs=7, rounds=3, seed=42)
        for round_index in range(1, 4):
            jobs = [job for job in schedule if job["round"] == round_index]
            self.assertCountEqual([(job["algorithm"], job["mode"]) for job in jobs], self.configurations)
            self.assertEqual([job["position"] for job in jobs], list(range(1, 7)))
        totals = Counter()
        for job in schedule:
            totals[job["algorithm"], job["mode"]] += job["measured_runs"]
        self.assertEqual(dict(totals), {configuration: 7 for configuration in self.configurations})

    def test_seed_reproduces_order_and_rounds_change_order(self):
        schedule = bench.build_schedule(self.configurations, 30, 10, 42)
        self.assertEqual(schedule, bench.build_schedule(self.configurations, 30, 10, 42))
        orders = {tuple((job["algorithm"], job["mode"]) for job in schedule if job["round"] == index)
                  for index in range(1, 11)}
        self.assertGreater(len(orders), 1)

    def test_fewer_samples_than_rounds(self):
        schedule = bench.build_schedule(self.configurations, 2, 10, 42)
        self.assertEqual(len(schedule), 12)
        self.assertEqual({job["round"] for job in schedule}, {1, 2})
        self.assertTrue(all(job["measured_runs"] == 1 for job in schedule))

    def test_no_shuffle_preserves_configured_order_in_every_round(self):
        configurations = list(reversed(self.configurations))
        schedule = bench.build_schedule(configurations, 7, 3, 42, shuffle=False)
        for round_index in range(1, 4):
            jobs = [job for job in schedule if job["round"] == round_index]
            self.assertEqual([(job["algorithm"], job["mode"]) for job in jobs], configurations)
        totals = Counter()
        for job in schedule:
            totals[job["algorithm"], job["mode"]] += job["measured_runs"]
        self.assertEqual(dict(totals), {configuration: 7 for configuration in configurations})

    def test_both_lifecycles_are_separate_configurations(self):
        configurations = [(algorithm, mode, lifecycle) for algorithm, mode in self.configurations
                          for lifecycle in bench.LIFECYCLES]
        schedule = bench.build_schedule(configurations, 5, 2, 42)
        totals = Counter()
        for job in schedule:
            totals[job["algorithm"], job["mode"], job["lifecycle"]] += job["measured_runs"]
        self.assertEqual(dict(totals), {configuration: 5 for configuration in configurations})
        for round_index in (1, 2):
            self.assertEqual(len([job for job in schedule if job["round"] == round_index]), 12)


class RoundAggregationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"warmup_runs": 3, "measured_runs": 2,
                         "timing_enabled": 1, "opencv_threads": 1}
        self.job = {"round": 1, "position": 4, "mode": "sequential"}
        self.samples = [{"total_ms": value, **{stage: value / 5 for stage in bench.STAGES}}
                        for value in (1.0, 3.0)]

    def test_pooled_statistics_and_round_tags(self):
        experiment = bench.merge_batch(None, self.metadata, self.samples, self.job, None, "result.png")
        job = {**self.job, "round": 2, "position": 1}
        samples = [{"total_ms": value, **{stage: value / 5 for stage in bench.STAGES}} for value in (5.0, 7.0)]
        experiment = bench.merge_batch(experiment, self.metadata, samples, job, None, "result.png")
        self.assertEqual(experiment["metadata"]["measured_runs"], 4)
        self.assertEqual(experiment["metadata"]["warmup_runs"], 6)
        self.assertEqual(experiment["statistics"]["median_ms"], 4.0)
        self.assertEqual(experiment["statistics"]["p95_ms"], 7.0)
        self.assertEqual([sample["round"] for sample in experiment["samples"]], [1, 1, 2, 2])
        self.assertEqual([sample["position"] for sample in experiment["samples"]], [4, 4, 1, 1])
        self.assertEqual([sample["iteration_in_round"] for sample in experiment["samples"]], [1, 2, 1, 2])

    def test_early_reference_failure_is_not_hidden(self):
        failure = {"match": False, "differing_pixels": 1, "difference_percent": 25.0}
        success = {"match": True, "differing_pixels": 0, "difference_percent": 0.0}
        experiment = bench.merge_batch(None, self.metadata, self.samples, self.job, failure, "result.png")
        experiment = bench.merge_batch(experiment, self.metadata, self.samples,
                                      {**self.job, "round": 2}, success, "result.png")
        self.assertFalse(experiment["comparison"]["match"])
        self.assertTrue(experiment["rounds"][1]["comparison"]["match"])

    def test_thread_count_change_rejected(self):
        experiment = bench.merge_batch(None, self.metadata, self.samples, self.job, None, "result.png")
        with self.assertRaisesRegex(ValueError, "opencv_threads"):
            bench.merge_batch(experiment, {**self.metadata, "opencv_threads": 2}, self.samples,
                              {**self.job, "round": 2}, None, "result.png")


@unittest.skipIf(Image is None, "Pillow is needed for image comparison tests")
class ComparisonTests(unittest.TestCase):
    def test_equal_pixels(self):
        image = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
        comparison = bench.compare_images(image, image.copy())
        self.assertTrue(comparison["match"])
        self.assertEqual(comparison["differing_pixels"], 0)

    def test_alpha_only_difference(self):
        expected = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
        actual = expected.copy()
        actual.putpixel((1, 1), (10, 20, 30, 0))
        comparison = bench.compare_images(actual, expected)
        self.assertEqual(comparison["differing_pixels"], 1)
        self.assertEqual(comparison["difference_percent"], 25.0)

    def test_hidden_rgb_is_compared(self):
        expected = Image.new("RGBA", (1, 1), (10, 20, 30, 0))
        actual = Image.new("RGBA", (1, 1), (50, 60, 70, 0))
        self.assertEqual(bench.compare_images(actual, expected)["differing_pixels"], 1)

    def test_count_pixels_not_channels(self):
        expected = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
        actual = expected.copy()
        actual.putpixel((0, 0), (255, 255, 255, 0))
        self.assertEqual(bench.compare_images(actual, expected)["differing_pixels"], 1)

    def test_dimensions_must_match(self):
        comparison = bench.compare_images(Image.new("RGBA", (1, 1)), Image.new("RGBA", (2, 1)))
        self.assertFalse(comparison["match"])
        self.assertEqual(comparison["reason"], "size mismatch")
        self.assertIsNone(comparison["difference_percent"])


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "schema_version": 1,
            "metadata": {"algorithm_requested": "base_bbdt", "algorithm_effective": "base_sauf",
                         "connectivity": 4, "parallel_requested": 0,
                         "measured_runs": 1, "warmup_runs": 0,
                         "debug_enabled": 0, "timing_enabled": 1, "build_configuration": "Release"},
            "samples": [{"total_ms": 5.0, **{stage: 1.0 for stage in bench.STAGES}}],
        }

    def validate(self):
        return bench.validate_measurements(self.data, "base_bbdt", "sequential", 1, 0)

    def test_bbdt_fallback_is_recorded(self):
        self.assertEqual(self.validate()[0]["algorithm_effective"], "base_sauf")

    def test_wrong_sample_count(self):
        self.data["samples"] = []
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonfinite_and_negative_values(self):
        for invalid in (-1, float("nan"), float("inf")):
            self.data["samples"][0]["total_ms"] = invalid
            with self.assertRaises(ValueError):
                self.validate()

    def test_debug_build_rejected(self):
        self.data["metadata"]["debug_enabled"] = 1
        with self.assertRaisesRegex(ValueError, "debugging disabled"):
            self.validate()

    def test_missing_stage_rejected(self):
        del self.data["samples"][0]["ccl_ms"]
        with self.assertRaises(KeyError):
            self.validate()

    def test_fresh_requires_matching_lifecycle_and_cleanup(self):
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            bench.validate_measurements(self.data, "base_bbdt", "sequential", 1, 0, "fresh")
        self.data["metadata"]["lifecycle"] = "fresh"
        with self.assertRaises(KeyError):
            bench.validate_measurements(self.data, "base_bbdt", "sequential", 1, 0, "fresh")
        self.data["samples"][0]["teardown_ms"] = 1.0
        bench.validate_measurements(self.data, "base_bbdt", "sequential", 1, 0, "fresh")
        self.data["samples"][0]["teardown_ms"] = 6.0
        with self.assertRaisesRegex(ValueError, "cleanup"):
            bench.validate_measurements(self.data, "base_bbdt", "sequential", 1, 0, "fresh")

    def test_reuse_cannot_include_instance_destruction(self):
        self.data["samples"][0]["teardown_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "Reuse"):
            self.validate()


@unittest.skipIf(Image is None, "Pillow is needed for integration fixtures")
class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.exe = ROOT / "build" / ("colorist.exe" if sys.platform == "win32" else "colorist")
        if not self.exe.is_file():
            self.skipTest("Build colorist before running integration tests")
        build = (ROOT / "build").resolve()
        self.assertEqual(build.parent, ROOT.resolve())
        self.temporary = tempfile.TemporaryDirectory(prefix="bench test ", dir=build)
        self.directory = Path(self.temporary.name)
        self.assertEqual(self.directory.resolve().parent, build)
        self.addCleanup(self.temporary.cleanup)
        Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(self.directory / "mask.png")
        Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(self.directory / "color.png")
        self.arguments = [str(self.exe), "--mask", str(self.directory / "mask.png"),
                          "--color_mask", str(self.directory / "color.png"),
                          "--output", str(self.directory / "result.png")]

    def test_repetitions_and_warmup_exclusion(self):
        metrics_path = self.directory / "metrics.json"
        completed = subprocess.run(self.arguments + ["--warmup", "2", "--runs", "5", "--parallel",
                                   "--metrics", str(metrics_path)], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with metrics_path.open() as stream:
            data = json.load(stream)
        self.assertEqual(len(data["samples"]), 5)
        self.assertEqual(data["metadata"]["warmup_runs"], 2)
        self.assertEqual(data["metadata"]["measured_runs"], 5)
        self.assertTrue(bench.compare_images(bench.load_rgba(self.directory / "result.png"),
                                            bench.load_rgba(self.directory / "color.png"))["match"])
        self.assertTrue(all(sample["total_ms"] >= 0 for sample in data["samples"]))

    def test_invalid_counts_and_algorithm_fail_before_writing(self):
        for arguments in (["--runs", "0"], ["--warmup", "-1"], ["--runs", "3oops"],
                          ["--runs", "999999999999999"], ["--algo-type", "typo"],
                          ["--algo-type", "rle_provisional"], ["--algo-type", "rle_uniform"],
                          ["--algo-type", "rle_complete"],
                          ["--algo-type", "rle_stream"], ["--algo-type", "rle", "--color-batch", "2"],
                          ["--threads", "2"], ["--parallel", "--threads", "0"],
                          ["--parallel", "--threads", "-1"], ["--parallel", "--threads", "3oops"],
                          ["--parallel", "--threads", "999999999999999"],
                          ["--parallel", "--threads", "2", "--threads", "4"],
                          ["--lifecycle", "typo"], ["--lifecycle", "fresh", "--lifecycle", "reuse"]):
            completed = subprocess.run(self.arguments + arguments, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((self.directory / "result.png").exists())

    def test_rle_parallel_thread_selection_and_one_thread_fallback(self):
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                metrics = self.directory / "threads.json"
                completed = subprocess.run(self.arguments + ["--algo-type", "rle", "--parallel",
                                           "--threads", str(threads), "--metrics", str(metrics)],
                                           capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                metadata = json.loads(metrics.read_text())["metadata"]
                self.assertEqual(metadata["opencv_threads"], threads)
                self.assertEqual(metadata["parallel_labeling"], int(threads > 1))
                self.assertEqual(metadata["parallel_coloring"], int(threads > 1))
                self.assertEqual(metadata["coloring_backend"], metadata["parallel_backend"] if threads > 1 else "sequential")
                self.assertTrue(bench.compare_images(bench.load_rgba(self.directory / "result.png"),
                                                    bench.load_rgba(self.directory / "color.png"))["match"])

    def test_metrics_cannot_overwrite_input(self):
        color = self.directory / "color.png"
        original = color.read_bytes()
        completed = subprocess.run(self.arguments + ["--metrics", str(color)], capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(color.read_bytes(), original)

    def test_runner_protects_reference(self):
        reference = self.directory / "base_spaghetti-sequential.png"
        Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(reference)
        original = reference.read_bytes()
        completed = subprocess.run([sys.executable, str(ROOT / "bench.py"), "--exe", str(self.exe),
                                   "--mask", str(self.directory / "mask.png"),
                                   "--color-mask", str(self.directory / "color.png"),
                                   "--reference", str(reference), "--output-dir", str(self.directory)],
                                  capture_output=True)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(reference.read_bytes(), original)

    def test_shuffled_runner_rounds_and_csv_counts(self):
        directory = self.directory / "matrix"
        completed = subprocess.run([sys.executable, str(ROOT / "bench.py"), "--exe", str(self.exe),
                                   "--mask", str(self.directory / "mask.png"),
                                   "--color-mask", str(self.directory / "color.png"),
                                   "--reference", str(self.directory / "color.png"),
                                   "--output-dir", str(directory), "--warmup", "1", "--runs", "5",
                                   "--rounds", "2", "--seed", "42", "--lifecycles", "reuse"], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with (directory / "report.json").open() as stream:
            report = json.load(stream)
        self.assertEqual(report["shuffle_seed"], 42)
        self.assertEqual(len(report["execution_order"]), 12)
        self.assertEqual(len(report["experiments"]), 6)
        for experiment in report["experiments"]:
            self.assertEqual(len(experiment["samples"]), 5)
            self.assertEqual(len(experiment["rounds"]), 2)
            self.assertEqual(experiment["metadata"]["warmup_runs"], 2)
            self.assertEqual(experiment["metadata"]["measured_runs"], 5)
            self.assertTrue(experiment["comparison"]["match"])
            name = f"{experiment['metadata']['algorithm_requested']}-{experiment['mode']}.json"
            with (directory / name).open() as stream:
                self.assertEqual(len(json.load(stream)["samples"]), 5)
        import csv
        with (directory / "measurements.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        self.assertEqual({row["round"] for row in rows}, {"1", "2"})

    def test_mixed_rle_runner_preserves_stage_names_and_parallel_coloring(self):
        directory = self.directory / "mixed"
        completed = subprocess.run([sys.executable, str(ROOT / "bench.py"), "--exe", str(self.exe),
                                   "--mask", str(self.directory / "mask.png"),
                                   "--color-mask", str(self.directory / "color.png"),
                                   "--reference", str(self.directory / "color.png"),
                                   "--output-dir", str(directory), "--algos", "base_sauf", "rle",
                                   "--threads", "2", "--warmup", "2", "--runs", "3",
                                   "--rounds", "2", "--no-shuffle"],
                                  capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with (directory / "report.json").open() as stream:
            report = json.load(stream)
        self.assertEqual(len(report["experiments"]), 8)
        self.assertEqual(report["lifecycles"], ["reuse", "fresh"])
        for experiment in report["experiments"]:
            metadata = experiment["metadata"]
            self.assertEqual(len(experiment["samples"]), 3)
            self.assertEqual(metadata["warmup_runs"], 4)
            self.assertTrue(experiment["comparison"]["match"])
            self.assertEqual(metadata["lifecycle"], experiment["lifecycle"])
            self.assertEqual(metadata["output_allocation_timed"], 0)
            self.assertEqual(metadata["output_buffer_reused"], 1)
            for sample in experiment["samples"]:
                self.assertGreaterEqual(sample["total_ms"], sample["teardown_ms"])
                if experiment["lifecycle"] == "reuse":
                    self.assertEqual(sample["teardown_ms"], 0)
            name = bench.experiment_name(metadata["algorithm_requested"], experiment["mode"], experiment["lifecycle"])
            self.assertTrue((directory / f"{name}.png").is_file())
            self.assertTrue((directory / f"{name}.json").is_file())
            if metadata["algorithm_requested"] in bench.RLE_ALGORITHMS:
                self.assertEqual(metadata["algorithm_effective"], metadata["algorithm_requested"])
                self.assertEqual(metadata["parallel_labeling"], int(experiment["mode"] == "parallel"))
                self.assertEqual(metadata["opencv_threads"], 2 if experiment["mode"] == "parallel" else 1)
                self.assertEqual(metadata["parallel_coloring"], int(experiment["mode"] == "parallel"))
                self.assertEqual(bench.execution_label(experiment),
                                 experiment["mode"])
                self.assertEqual(metadata["foreground_threshold"], 127)
            expected_stages = bench.timing_stages(metadata) if metadata["timing_enabled"] else ()
            self.assertEqual(set(experiment["stage_statistics"]), set(expected_stages))
        import csv
        with (directory / "measurements.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["lifecycle"] for row in rows}, {"reuse", "fresh"})
        if report["experiments"][0]["metadata"]["timing_enabled"]:
            for row in rows:
                self.assertEqual(bool(row["scan_ccl_histogram_ms"]), row["algorithm"] in bench.RLE_ALGORITHMS)
                self.assertEqual(bool(row["ccl_ms"]), row["algorithm"] not in bench.RLE_ALGORITHMS)


if __name__ == "__main__":
    unittest.main()
