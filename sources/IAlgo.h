#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class IAlgo {
public:
    virtual ~IAlgo() = default;
    virtual cv::Mat Compute(const cv::Mat& mask, const cv::Mat& color_mask) = 0;
};
