#ifdef COLORIST_DEBUG
#include <opencv2/core/check.hpp>
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include "BaseAlgo.h"
#include "Benchmark.h"
#include "RleAlgo.h"

CclAlgoType GetCclAlgoType(const std::string& ccl_algo) {
    if (ccl_algo == "base_spaghetti") {
        return CclAlgoType::CCL_SPAGHETTI;
    } else if (ccl_algo == "base_sauf") {
        return CclAlgoType::CCL_SAUF;
    } else if (ccl_algo == "base_bbdt") {
        return CclAlgoType::CCL_BBDT;
    } else {
        throw std::runtime_error("Unknown algorithm: " + ccl_algo);
    }
}

std::unique_ptr<IAlgo> CreateAlgo(const std::string& name, int num_threads = 1) {
    if (name == "rle") {
        return std::make_unique<RleAlgo>(num_threads);
    }

    auto algo = std::make_unique<BaseAlgo>();
    algo->SetAlgoType(GetCclAlgoType(name));
    return algo;
}

int ParseCount(const std::string& value, const char* option, int minimum) {
    int count = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || count < minimum)
        throw std::runtime_error(std::string(option) + " must be an integer >= " + std::to_string(minimum));
    return count;
}

void SaveImage(const std::string& filename, const cv::Mat& image) {
    const std::filesystem::path destination(filename);
    auto temporary = destination;
    temporary += ".tmp-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + destination.extension().string();

    try {
        errno = 0;
        if (!cv::imwrite(temporary.string(), image)) {
            const int write_error = errno;
            throw std::runtime_error("Cannot save " + filename + ": " +
                (write_error ? std::generic_category().message(write_error) : "image encoding failed"));
        }
        std::error_code error;
        std::filesystem::rename(temporary, destination, error);
        if (error) {
            throw std::runtime_error("Cannot replace " + filename + ": " + error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

#ifdef COLORIST_DEBUG
void ShowImageInfo(const cv::Mat& img) {
    std::cout << "Type: " << cv::typeToString(img.type()) << " (" << img.type() << ")"
              << " -- Depth: " << cv::depthToString(img.depth()) << " (" << img.depth() << ")"
              << " -- Channels: " << img.channels() << '\n';
}
#endif

int main(int argc, char* argv[]) {
    constexpr auto usage =
        "Usage: colorist --mask <mask.png> --color_mask <color_mask.png> --output <result.png>\n"
        "                [--algo-type <base_spaghetti|base_sauf|base_bbdt|rle>] [--parallel [--threads <count>]]\n"
        "                [--lifecycle <reuse|fresh>] [--warmup <count>] [--runs <count>]\n"
        "                [--metrics <measurements.json>]\n";

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << usage;
        return 0;
    }

    try {
        std::string mask_path, color_mask_path, output_path, algo_type;
        std::string warmup_arg, runs_arg, metrics_path, lifecycle, threads_arg;
        bool parallel = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--parallel") {
                if (parallel) {
                    throw std::runtime_error("Duplicate argument: --parallel");
                }
                parallel = true;
                continue;
            }
            std::string* path = option == "--mask" ? &mask_path
                              : option == "--color_mask" ? &color_mask_path
                              : option == "--output" ? &output_path
                              : option == "--algo-type" ? &algo_type
                              : option == "--warmup" ? &warmup_arg
                              : option == "--runs" ? &runs_arg
                              : option == "--metrics" ? &metrics_path
                              : option == "--lifecycle" ? &lifecycle
                              : option == "--threads" ? &threads_arg
                              : nullptr;
            if (!path) {
                throw std::runtime_error("Unknown argument: " + option);
            }
            if (i + 1 == argc) {
                throw std::runtime_error("Missing value for " + option);
            }
            if (!path->empty()) {
                throw std::runtime_error("Duplicate argument: " + option);
            }
            *path = argv[++i];
            if (path->empty()) {
                throw std::runtime_error("Missing value for " + option);
            }
        }
        if (mask_path.empty() || color_mask_path.empty() || output_path.empty()) {
            throw std::runtime_error("--mask, --color_mask and --output are required");
        }
        if (algo_type.empty()) {
            algo_type = "base_spaghetti";
        }
        auto algo = CreateAlgo(algo_type);
        if (lifecycle.empty())
            lifecycle = "reuse";
        if (lifecycle != "reuse" && lifecycle != "fresh")
            throw std::runtime_error("--lifecycle must be reuse or fresh");
        const bool fresh = lifecycle == "fresh";
        const int warmup_runs = warmup_arg.empty() ? 0 : ParseCount(warmup_arg, "--warmup", 0);
        const int measured_runs = runs_arg.empty() ? 1 : ParseCount(runs_arg, "--runs", 1);
        const int requested_threads = threads_arg.empty() ? -1 : ParseCount(threads_arg, "--threads", 1);
        if (!threads_arg.empty() && !parallel)
            throw std::runtime_error("--threads requires --parallel");
        const auto absolute_path = [](const std::string& path) {
            return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
        };
        if (absolute_path(output_path) == absolute_path(mask_path) ||
            absolute_path(output_path) == absolute_path(color_mask_path)) {
            throw std::runtime_error("Output must differ from the input files");
        }
        if (!metrics_path.empty() && (absolute_path(metrics_path) == absolute_path(mask_path) ||
            absolute_path(metrics_path) == absolute_path(color_mask_path) ||
            absolute_path(metrics_path) == absolute_path(output_path))) {
            throw std::runtime_error("Measurements must differ from the input and output files");
        }

        const cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            throw std::runtime_error("Cannot read mask: " + mask_path);
        }
#ifdef COLORIST_DEBUG
        ShowImageInfo(mask);
#endif

        const cv::Mat color_mask = cv::imread(color_mask_path, cv::IMREAD_UNCHANGED);
        if (color_mask.empty()) {
            throw std::runtime_error("Cannot read color_mask: " + color_mask_path);
        }
#ifdef COLORIST_DEBUG
        ShowImageInfo(color_mask);

        // Save the input preview before running the algorithm, outside its timer.
        {
            cv::Mat overlay;
            cv::cvtColor(mask, overlay, cv::COLOR_GRAY2BGRA);
            cv::addWeighted(color_mask, 0.5, overlay, 0.5, 0.0, overlay);
            SaveImage("overlay.png", overlay);
            std::cout << "Saved overlay.png\n";
        }
#endif

        const bool supports_parallel = algo->SupportsParallelLabeling();
        const bool stripe_labeling = algo_type == "rle";
        const bool supports_parallel_coloring = algo->SupportsParallelColoring();
        cv::setNumThreads(parallel && (supports_parallel || supports_parallel_coloring) ? requested_threads : 1);
        const int thread_count = cv::getNumThreads();
        algo->SetNumThreads(thread_count);
        const int rows_per_thread = thread_count > 0 ? mask.rows / thread_count : 0;
        const bool enough_rows = thread_count > 0 && rows_per_thread >= 2;
        const bool parallel_labeling = supports_parallel && parallel && thread_count > 1 &&
            (stripe_labeling ? mask.rows > 1 : enough_rows);
        const bool parallel_coloring = supports_parallel_coloring && parallel && thread_count > 1 && mask.rows > 1;
        std::cout << "Parallel requested: " << (parallel ? "yes" : "no")
                  << " -- OpenCV threads: " << thread_count << '\n';
        if (stripe_labeling) {
            std::cout << "RLE stripes: " << std::min(mask.rows, thread_count)
                      << " -- boundary reconciliation: sequential\n";
        } else if (thread_count > 0) {
            std::cout << "Row check: image.rows / cv::getNumThreads() >= 2: "
                      << mask.rows << " / " << thread_count << " = " << rows_per_thread
                      << " >= 2 -> " << (enough_rows ? "PASS" : "FAIL") << '\n';
        } else {
            std::cout << "Row check: unavailable (zero OpenCV threads)\n";
        }
        std::cout << "Connected components: "
                  << (parallel_labeling ? "parallel version running" : "sequential version running")
                  << '\n';
        std::cout << "Final coloring: "
                  << (parallel_coloring ? "parallel version running" : "sequential version running")
                  << (parallel_coloring ? " (OpenCV parallel_for_)" : "")
                  << '\n';

        if (warmup_runs > 0 || measured_runs > 1) {
            std::cout << "Warm-up runs: " << warmup_runs << " -- measured runs: " << measured_runs << '\n';
        }
        std::cout << "Lifecycle: " << lifecycle
                  << (fresh ? " (Compute + instance destruction timed)\n" : " (Compute including reset timed)\n");
        // Allocate once outside all timers. Each algorithm paints every pixel on every call.
        cv::Mat result(mask.size(), algo->GetOutputType());
        for (int run = 0; run < warmup_runs; ++run) {
            auto warmup_algo = fresh ? CreateAlgo(algo_type, thread_count) : nullptr;
            (warmup_algo ? warmup_algo.get() : algo.get())->Compute(mask, color_mask, result);
        }

        std::vector<Measurement> measurements;
        measurements.reserve(measured_runs);
        for (int run = 0; run < measured_runs; ++run) {
            auto fresh_algo = fresh ? CreateAlgo(algo_type, thread_count) : nullptr;
            auto* current_algo = fresh_algo ? fresh_algo.get() : algo.get();
            const auto start = std::chrono::steady_clock::now();
            current_algo->Compute(mask, color_mask, result);
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            Measurement measurement;
            measurement.total_ms = elapsed_ms;
#ifdef COLORIST_TIMING
            measurement.stages_ms = current_algo->GetTimings();
#endif
            if (fresh_algo) {
                // The caller-owned output stays alive; time algorithm instance/workspace destruction.
                const auto teardown_start = std::chrono::steady_clock::now();
                fresh_algo.reset();
                measurement.teardown_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - teardown_start).count();
                measurement.total_ms += measurement.teardown_ms;
            }
            measurements.push_back(measurement);
        }
        const double mean_ms = std::accumulate(measurements.begin(), measurements.end(), 0.0,
            [](double sum, const Measurement& sample) { return sum + sample.total_ms; }) / measured_runs;
        std::cout << "Algorithm time" << (measured_runs > 1 ? " (mean)" : "") << ": " << mean_ms << " ms\n";
#ifdef COLORIST_TIMING
        if (measured_runs > 1)
            std::cout << "Stage timings for the last measured run:\n";
        algo->PrintTimings(measurements.back().stages_ms);
#endif
        if (fresh)
            std::cout << "Instance cleanup (last measured run): " << measurements.back().teardown_ms << " ms\n";

        SaveImage(output_path, result);
        std::cout << "Saved " << output_path << '\n';
        if (!metrics_path.empty()) {
            SaveMeasurements(metrics_path, algo_type, parallel, parallel_labeling, parallel_coloring, thread_count,
                             mask, BaseAlgo::Connectivity, warmup_runs, measurements, *algo, lifecycle);
            std::cout << "Saved " << metrics_path << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n' << usage;
        return 1;
    }
}
