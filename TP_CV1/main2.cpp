#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int disp12MaxDiff = 1;
Mat gRectified1;
Mat gRectified2;

void findMatchings(Mat& image1, Mat& image2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
	std::vector<cv::Point2f> tmpA, tmpB;
	std::vector<unsigned char> status;
	std::vector<float> errors;
	goodFeaturesToTrack(image1, tmpA, 5000, 0.01, 10.0f);
	calcOpticalFlowPyrLK(image1, image2, tmpA, tmpB, status, errors);
	auto tmpABegin = tmpA.begin(), tmpBBegin = tmpB.begin();
	auto statusBegin = status.begin();
	while (tmpABegin != tmpA.end())
	{
		if ( (*statusBegin) == 1)
		{
			points1.push_back(*tmpABegin);
			points2.push_back(*tmpBBegin);
		}
		tmpABegin++;
		tmpBBegin++;
		statusBegin++;
	}
	std::cout << "Size :" << points1.size() << std::endl;
}


void showMatchings(Mat& image1, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, int nb)
{
	Size sz1 = image1.size();
	Mat im3(sz1.height, sz1.width, image1.type());
	image1.copyTo(im3);
	Mat im3Color;
	cv::cvtColor(im3, im3Color, CV_GRAY2BGR);
	auto p1Ptr = points1.begin(), p2Ptr = points2.begin();
	while (p1Ptr != points1.end())
	{
		cv::line(im3Color, *p1Ptr, *p2Ptr, cv::Scalar(255.0, 0.0, 0.0));
		p1Ptr++;
		p2Ptr++;
	}
	imshow(std::string("Image")+std::to_string(nb), im3Color);
}

void rectify(Mat& image1, Mat& image2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, Mat& rectified1, Mat& rectified2)
{
	Mat fund = findFundamentalMat(points1, points2);
	Mat h1, h2;
	stereoRectifyUncalibrated(points1, points2, fund, image1.size(), h1, h2);
	warpPerspective(image1, rectified1, h1, image1.size());
	warpPerspective(image2, rectified2, h2, image2.size());
}



void findMatrix(Mat& image1, Mat& image2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
	Mat fund = findFundamentalMat(points1, points2);
}

Mat computeDisparity()
{
	Mat dispatiry, dispatiry8U;
	Ptr<StereoBM> sbm = StereoBM::create();

	sbm->setDisp12MaxDiff(disp12MaxDiff);
	/*
	sbm->setSpeckleRange(8);
	sbm->setSpeckleWindowSize(0);
	sbm->setUniquenessRatio(0);
	sbm->setTextureThreshold(507);
	sbm->setMinDisparity(-39);
	sbm->setPreFilterCap(61);
	sbm->setPreFilterSize(5);*/

	sbm->compute(gRectified1, gRectified2, dispatiry);
	double minVal; double maxVal;
	minMaxLoc(dispatiry, &minVal, &maxVal);
	printf("Min disp: %f Max value: %f \n", minVal, maxVal);
	dispatiry.convertTo(dispatiry8U, CV_8UC1, 255 / (maxVal - minVal));

	cv::imshow("Disparity", dispatiry8U);
	return dispatiry8U;
}

void updateDisparity(int v, void* val)
{
	disp12MaxDiff = v;
	computeDisparity();
}

int main(int argc, char* argv[])
{
	for (int nb_img = 100; nb_img < 120; nb_img++)
	{
		cv::Mat image1, image2;
		std::string name1, name2;
		if (nb_img < 10)
			name1 = std::string("images/000000000") + std::to_string(nb_img) + std::string(".png");
		else if (nb_img < 100)
			name1 = std::string("images/00000000") + std::to_string(nb_img) + std::string(".png");
		else if (nb_img < 1000)
			name1 = std::string("images/0000000") + std::to_string(nb_img) + std::string(".png");
		else
			name1 = std::string("images/000000") + std::to_string(nb_img) + std::string(".png");
		if (nb_img < 9)
			name2 = std::string("images/000000000") + std::to_string(nb_img + 1) + std::string(".png");
		else if (nb_img < 99)
			name2 = std::string("images/00000000") + std::to_string(nb_img + 1) + std::string(".png");
		else if (nb_img < 999)
			name2 = std::string("images/0000000") + std::to_string(nb_img + 1) + std::string(".png");
		else
			name2 = std::string("images/000000") + std::to_string(nb_img + 1) + std::string(".png");
		image1 = cv::imread(name1, CV_LOAD_IMAGE_GRAYSCALE);
		image2 = cv::imread(name2, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<cv::Point2f> points1, points2;
		findMatchings(image1, image2, points1, points2);
		findMatchings(image2, image1, points2, points1);
		showMatchings(image1, points1, points2, nb_img);
		findMatrix(image1, image2, points1, points2);
	}
	std::cout << "end" << std::endl;
	cv::waitKey();
	cv::waitKey();
	return 0;
}