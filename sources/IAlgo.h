#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef COLORIST_TIMING
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

class IAlgo {
public:
    virtual ~IAlgo() = default;
    virtual int GetOutputType() const { return CV_8UC4; }
    // Pixel type the color mask is decoded to (outside timing): BGRA, or BGR when alpha is unused.
    virtual int GetColorMaskType() const { return CV_8UC4; }
    // Caller provides GetOutputType() storage of the input size; every pixel is overwritten.
    virtual void Compute(const cv::Mat& mask, const cv::Mat& color_mask, cv::Mat& output) = 0;
    virtual bool SupportsParallelLabeling() const { return false; }
    virtual bool SupportsParallelColoring() const { return false; }
    virtual void SetNumThreads(int) {}
#ifdef COLORIST_TIMING
    virtual const std::array<double, 5>& GetTimings() const = 0;
    virtual const std::array<const char*, 5>& GetTimingKeys() const = 0;  // Metrics JSON keys.
    virtual const std::array<const char*, 5>& GetStageNames() const = 0;  // Table labels.

    void PrintTimings(const std::array<double, 5>& timings) const {
        const char* border = "+------------------+-----------+\n";
        std::ostringstream table;
        table << '\n' << border << "| Stage            |        ms |\n" << border
              << std::fixed << std::setprecision(3);
        for (std::size_t i = 0; i < timings.size(); ++i)
            table << "| " << std::left << std::setw(16) << GetStageNames()[i]
                  << " | " << std::right << std::setw(9) << timings[i] << " |\n";
        table << border;
        std::cout << table.str();
    }
#endif
};
