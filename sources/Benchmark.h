#pragma once

#include "IAlgo.h"

#include <opencv2/core.hpp>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Measurement
{
    double total_ms{};
    double teardown_ms{};
#ifdef COLORIST_TIMING
    std::array<double, 5> stages_ms{};
#endif
};

inline void SaveMeasurements(const std::string& filename, const std::string& algorithm,
                             bool parallel_requested, bool parallel_labeling, bool parallel_coloring, int threads,
                             const cv::Mat& mask, int connectivity, int warmup_runs,
                             const std::vector<Measurement>& measurements, const IAlgo& algo,
                             const std::string& lifecycle)
{
    cv::FileStorage file(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened())
        throw std::runtime_error("Cannot write measurements: " + filename);

    const std::string build_info = cv::getBuildInformation();
    std::string backend = "unknown";
    std::istringstream lines(build_info);
    for (std::string line; std::getline(lines, line);)
    {
        const auto position = line.find("Parallel framework:");
        if (position != std::string::npos)
        {
            backend = line.substr(position + std::string("Parallel framework:").size());
            backend.erase(0, backend.find_first_not_of(" \t"));
            break;
        }
    }

    file << "schema_version" << 1 << "metadata" << "{"
         << "algorithm_requested" << algorithm
         << "lifecycle" << lifecycle
         << "output_allocation_timed" << 0 << "output_buffer_reused" << 1
         << "algorithm_effective" << (algorithm == "base_bbdt" && connectivity == 4 ? "base_sauf" : algorithm)
         << "parallel_requested" << int(parallel_requested)
         << "parallel_labeling" << int(parallel_labeling)
         << "parallel_coloring" << int(parallel_coloring)
         << "coloring_backend" << (parallel_coloring ? backend : "sequential")
         << "opencv_threads" << threads << "parallel_backend" << backend
         << "connectivity" << connectivity << "image_width" << mask.cols << "image_height" << mask.rows
         << "foreground_threshold" << 127
         << "warmup_runs" << warmup_runs << "measured_runs" << int(measurements.size())
         << "opencv_version" << CV_VERSION
         << "build_configuration" << COLORIST_BUILD_CONFIG << "compiler" << COLORIST_COMPILER
#ifdef COLORIST_TIMING
         << "timing_enabled" << 1
#else
         << "timing_enabled" << 0
#endif
#ifdef COLORIST_DEBUG
         << "debug_enabled" << 1;
#else
         << "debug_enabled" << 0;
#endif
#ifdef COLORIST_TIMING
    file << "stage_keys" << "[";
    for (const auto key : algo.GetTimingKeys())
        file << key;
    file << "]";
#endif
    file << "opencv_build_information" << "[";
    // FileStorage limits individual strings; retain the full build report as lines.
    std::istringstream build_lines(build_info);
    for (std::string line; std::getline(build_lines, line);)
        file << line;
    file << "]" << "}" << "samples" << "[";
    for (const auto& measurement : measurements)
    {
        file << "{" << "total_ms" << measurement.total_ms << "teardown_ms" << measurement.teardown_ms;
#ifdef COLORIST_TIMING
        for (std::size_t stage = 0; stage < measurement.stages_ms.size(); ++stage)
            file << algo.GetTimingKeys()[stage] << measurement.stages_ms[stage];
#endif
        file << "}";
    }
    file << "]";
    file.release();
}
