#ifndef ALL_INCLUDES_VISION
#define ALL_INCLUDES_VISION

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

void updateMinH(int v, void* val);
void updateMaxH(int v, void* val);

void updateMinS(int v, void* val);
void updateMaxS(int v, void* val);

void updateMinV(int v, void* val);
void updateMaxV(int v, void* val);

void updateNBImage(int v, void* val);


//return the string corresponding to the correct folder
std::string getImagePrefix(int nbImage);

void getContoursAndMasks(const cv::Mat* baseImage, const cv::Scalar* rectColor, const cv::Scalar* circleColor, const cv::Scalar* triangleColor, std::vector<cv::Mat>& masks);

void initKnownDescriptors(std::string& prefix, cv::Ptr<cv::FeatureDetector> featureDetector, cv::Ptr<cv::DescriptorExtractor> descriptorExtractor, std::vector<cv::Mat>& knownDescriptors);

void getDescriptorAndDrawKeypoints(cv::Ptr<cv::FeatureDetector> featureDetector, cv::Ptr<cv::DescriptorExtractor> descriptorExtractor, std::vector<cv::Mat>& descriptors, cv::Mat image, std::vector<cv::Mat> masks);

#endif