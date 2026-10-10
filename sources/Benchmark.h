#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr const char* StageKeys[] = {
    "threshold_ms", "ccl_ms", "histogram_ms", "color_selection_ms", "final_coloring_ms"
};

struct Measurement
{
    double total_ms{};
#ifdef COLORIST_TIMING
    std::array<double, 5> stages_ms{};
#endif
};

inline void SaveMeasurements(const std::string& filename, const std::string& algorithm,
                             bool parallel_requested, bool parallel_labeling, int threads,
                             const cv::Mat& mask, int connectivity, int warmup_runs,
                             const std::vector<Measurement>& measurements)
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
         << "algorithm_effective" << (algorithm == "base_bbdt" && connectivity == 4 ? "base_sauf" : algorithm)
         << "parallel_requested" << int(parallel_requested)
         << "parallel_labeling" << int(parallel_labeling)
         << "opencv_threads" << threads << "parallel_backend" << backend
         << "connectivity" << connectivity << "image_width" << mask.cols << "image_height" << mask.rows
         << "warmup_runs" << warmup_runs << "measured_runs" << int(measurements.size())
         << "opencv_version" << CV_VERSION
         << "build_configuration" << COLORIST_BUILD_CONFIG << "compiler" << COLORIST_COMPILER
#ifdef COLORIST_TIMING
         << "timing_enabled" << 1
#else
         << "timing_enabled" << 0
#endif
#ifdef COLORIST_DEBUG
         << "debug_enabled" << 1
#else
         << "debug_enabled" << 0
#endif
         << "opencv_build_information" << "[";
    // FileStorage limits individual strings; retain the full build report as lines.
    std::istringstream build_lines(build_info);
    for (std::string line; std::getline(build_lines, line);)
        file << line;
    file << "]" << "}" << "samples" << "[";
    for (const auto& measurement : measurements)
    {
        file << "{" << "total_ms" << measurement.total_ms;
#ifdef COLORIST_TIMING
        for (std::size_t stage = 0; stage < measurement.stages_ms.size(); ++stage)
            file << StageKeys[stage] << measurement.stages_ms[stage];
#endif
        file << "}";
    }
    file << "]";
    file.release();
}
