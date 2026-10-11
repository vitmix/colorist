#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef COLORIST_TIMING
#include <array>
#endif

class IAlgo {
public:
    virtual ~IAlgo() = default;
    // Caller provides a CV_8UC4 output of the input size; every pixel is overwritten.
    virtual void Compute(const cv::Mat& mask, const cv::Mat& color_mask, cv::Mat& output) = 0;
    virtual bool SupportsParallelLabeling() const { return false; }
#ifdef COLORIST_TIMING
    virtual const std::array<double, 5>& GetTimings() const = 0;
    virtual const std::array<const char*, 5>& GetTimingKeys() const = 0;
    virtual void PrintTimings(const std::array<double, 5>& timings) const = 0;
    void PrintTimings() const { PrintTimings(GetTimings()); }
#endif
};
