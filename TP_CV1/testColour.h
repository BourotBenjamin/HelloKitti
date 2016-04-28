#ifndef TEST_COLOUR
#define TEST_COLOUR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

cv::Scalar dominantColour(cv::Mat& img);

bool compareColour(const cv::Scalar& col1, const cv::Scalar& col2, float precision = 0.2f);

#endif