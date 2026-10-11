#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef COLORIST_TIMING
#include <array>
#endif

class IAlgo {
public:
    virtual ~IAlgo() = default;
    virtual cv::Mat Compute(const cv::Mat& mask, const cv::Mat& color_mask) = 0;
    virtual bool SupportsParallelLabeling() const { return false; }
#ifdef COLORIST_TIMING
    virtual const std::array<double, 5>& GetTimings() const = 0;
    virtual const std::array<const char*, 5>& GetTimingKeys() const = 0;
    virtual void PrintTimings() const = 0;
#endif
};
