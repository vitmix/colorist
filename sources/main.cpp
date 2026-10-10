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
        "Usage: colorist --mask <mask.png> --color_mask <color_mask.png> --output <result.png>\n";

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << usage;
        return 0;
    }

    try {
        std::string mask_path, color_mask_path, output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            std::string* path = option == "--mask" ? &mask_path
                              : option == "--color_mask" ? &color_mask_path
                              : option == "--output" ? &output_path
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
        algo.SetAlgoType(cv::CCL_SPAGHETTI);

        const auto start = std::chrono::steady_clock::now();
        const cv::Mat result = algo.Compute(mask, color_mask);
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start);
        std::cout << "Algorithm time: " << elapsed.count() << " ms\n";

        SaveImage(output_path, result);
        std::cout << "Saved " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n' << usage;
        return 1;
    }
}
