#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

void processImageGPU(
    const cv::Mat& input,
    cv::Mat& grayscale,
    cv::Mat& blurred,
    cv::Mat& edges
);

void processImageCPU(
    const cv::Mat& input,
    cv::Mat& grayscale,
    cv::Mat& blurred,
    cv::Mat& edges
);

#endif
