#ifdef COLORIST_DEBUG
#include <opencv2/core/check.hpp>
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "BaseAlgo.h"

CclAlgoType GetCclAlgoType(const std::string& ccl_algo) {
    if (ccl_algo == "base_spaghetti") {
        return CclAlgoType::CCL_SPAGHETTI;
    } else if (ccl_algo == "base_sauf") {
        return CclAlgoType::CCL_SAUF;
    } else if (ccl_algo == "base_bbdt") {
        return CclAlgoType::CCL_BBDT;
    } else {
        return CclAlgoType::CCL_DEFAULT;
    }
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
        "                [--algo-type <base_spaghetti|base_sauf|base_bbdt>] [--parallel]\n";

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << usage;
        return 0;
    }

    try {
        std::string mask_path, color_mask_path, output_path, algo_type;
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
        }
        if (mask_path.empty() || color_mask_path.empty() || output_path.empty()) {
            throw std::runtime_error("--mask, --color_mask and --output are required");
        }
        if (algo_type.empty()) {
            algo_type = "base_spaghetti";
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

        BaseAlgo algo;
        algo.SetAlgoType(GetCclAlgoType(algo_type));

        cv::setNumThreads(parallel ? -1 : 1);
        const int thread_count = cv::getNumThreads();
        const int rows_per_thread = thread_count > 0 ? mask.rows / thread_count : 0;
        const bool enough_rows = thread_count > 0 && rows_per_thread >= 2;
        const bool parallel_labeling = parallel && thread_count > 1 && enough_rows;
        std::cout << "Parallel requested: " << (parallel ? "yes" : "no")
                  << " -- OpenCV threads: " << thread_count << '\n';
        if (thread_count > 0) {
            std::cout << "Row check: image.rows / cv::getNumThreads() >= 2: "
                      << mask.rows << " / " << thread_count << " = " << rows_per_thread
                      << " >= 2 -> " << (enough_rows ? "PASS" : "FAIL") << '\n';
        } else {
            std::cout << "Row check: unavailable (zero OpenCV threads)\n";
        }
        std::cout << "Connected components: "
                  << (parallel_labeling ? "parallel version running" : "sequential version running")
                  << '\n';

        const auto start = std::chrono::steady_clock::now();
        const cv::Mat result = algo.Compute(mask, color_mask);
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start);
        std::cout << "Algorithm time: " << elapsed.count() << " ms\n";
#ifdef COLORIST_TIMING
        algo.PrintTimings();
#endif

        SaveImage(output_path, result);
        std::cout << "Saved " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n' << usage;
        return 1;
    }
}
