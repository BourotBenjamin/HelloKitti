#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int minH, maxH = 255, minS, maxS = 255, minV, maxV = 255, nb_img = 0;
Mat gRectified1;
Mat gRectified2;

void findMatchings(Mat& image1, Mat& image2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
	std::vector<cv::Point2f> tmpA, tmpB;
	std::vector<unsigned char> status;
	std::vector<float> errors;
	goodFeaturesToTrack(image1, tmpA, 50000, 0.1, 10.0f);
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
	//std::cout << "Size :" << points1.size() << std::endl;
}


void showMatchings(Mat& image1, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, int nb)
{
	Size sz1 = image1.size();
	Mat im3(sz1.height, sz1.width, image1.type());
	image1.copyTo(im3);
	Mat im3Color;
	cv::cvtColor(im3, im3Color, CV_GRAY2BGR);
	/*int* aver = (int*) malloc(sizeof(int) * 2000);
	for (int i = 0; i < 2000; i++)
		aver[i] = 0;*/
	auto p1Ptr = points1.begin(), p2Ptr = points2.begin();
	while (p1Ptr != points1.end())
	{
		cv::line(im3Color, *p1Ptr, *p2Ptr, cv::Scalar(255.0, 0.0, 0.0));
		/*int x = p1Ptr->x / 100;
		int y = p1Ptr->y / 100;
		aver[x * 100 + y * 5] += p1Ptr->x;
		aver[x * 100 + y * 5 + 1] += p2Ptr->x;
		aver[x * 100 + y * 5 + 2] += p1Ptr->y;
		aver[x * 100 + y * 5 + 3] += p2Ptr->y;
		++aver[x * 100 + y * 5 + 4];*/
		p1Ptr++;
		p2Ptr++;
	}
	/*for (int i = 0; i < 2000; i += 5)
	{
		if (aver[i + 4] > 0)
		{
			cv::Point p1(aver[i] / aver[i + 4], aver[i + 2] / aver[i + 4]);
			cv::Point p2(aver[i + 1] / aver[i + 4], aver[i + 3] / aver[i + 4]);
			cv::line(im3Color, p1, p2, cv::Scalar(0.0, 0.0, 255.0));
		}
	}*/
	imshow(std::string("Image"), im3Color);
}



void updateMinH(int v, void* val)
{
	minH = v;
}
void updateMaxH(int v, void* val)
{
	maxH = v;
}

void updateMinS(int v, void* val)
{
	minS = v;
}
void updateMaxS(int v, void* val)
{
	maxS = v;
}

void updateMinV(int v, void* val)
{
	minV = v;
}
void updateMaxV(int v, void* val)
{
	maxV = v;
}


void updateNBImage(int v, void* val)
{
	nb_img = v;
}

std::string getImagePrefix(int i)
{
	if (nb_img < 10)
		return std::string("images/000000000");
	else if (nb_img < 100)
		return std::string("images/00000000");
	else if (nb_img < 1000)
		return std::string("images/0000000");
	else
		return std::string("images/000000");
}

void getContoursAndDraw(const Mat* baseImage, const Scalar* min, const Scalar* max, const Scalar* color, Mat* imageToDrawOn)
{
	Mat tmp;
	vector<vector<Point>> contours;
	inRange(*baseImage, *min, *max, tmp);
	findContours(tmp, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	int i = 0;
	for (const vector<Point>& cnt : contours)
	{
		if (contourArea(cnt) > 200)
			drawContours(*imageToDrawOn, contours, i, *color);
		i++;
	}
}

int main4(int argc, char* argv[])
{
	Scalar green(0, 255, 0);
	Mat image1, hsv, blue, blue2;
	Scalar blueMin(105, 181, 32), blueMax(115, 239, 214), blue2Min(107, 114, 32), blue2Max(110, 156, 44);
	for (nb_img = 0; nb_img < 837; nb_img++)
	{
		image1 = cv::imread(getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		cvtColor(image1, hsv, CV_BGR2HSV);
		getContoursAndDraw(&hsv, &blueMin, &blueMax, &green, &image1);
		getContoursAndDraw(&hsv, &blue2Min, &blue2Max, &green, &image1);
		cv::imshow("Color", image1);
		waitKey(1);
	}
	cv::waitKey();
	std::cout << "end" << std::endl;
	return 0;
}


int main3(int argc, char* argv[])
{
	Scalar green(0, 255, 0);
	Mat image1, hsv, tmp2, tmp3;
	Scalar min1(105, 181, 32), max1(115, 239, 214),
		min2(107, 114, 32), max2(110, 156, 44),
		min3(160, 86, 55), max3(184, 195, 93),
		min4(0, 150, 33), max4(5, 230, 50),
		min5(97, 76, 69), max5(106, 108, 89),
		min6(171, 82, 0), max6(197, 255, 47),
		min8(12, 172, 35), max8(21, 200, 61);
	for (nb_img = 0; nb_img < 837; nb_img++)
	{
		image1 = cv::imread(getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		cvtColor(image1, hsv, CV_BGR2HSV);

		vector<vector<Point>> contours;
		inRange(hsv, min1, max1, tmp3);
		inRange(hsv, min2, max2, tmp2);
		max(tmp3, tmp2, tmp3);
		inRange(hsv, min3, max3, tmp2);
		max(tmp3, tmp2, tmp3);
		inRange(hsv, min4, max4, tmp2);
		max(tmp3, tmp2, tmp3);
		inRange(hsv, min5, max5, tmp2);
		max(tmp3, tmp2, tmp3);
		inRange(hsv, min6, max6, tmp2);
		max(tmp3, tmp2, tmp3);
		inRange(hsv, min8, max8, tmp2);
		max(tmp3, tmp2, tmp3);
		cv::imshow("White", tmp3);
		findContours(tmp3, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		int i = 0;
		for (const vector<Point>& cnt : contours)
		{
			if (contourArea(cnt) > 200)
				drawContours(image1, contours, i, green);
			i++;
		}
		cv::imshow("Color", image1);
		waitKey(1);
	}
	cv::waitKey();
	std::cout << "end" << std::endl;
	return 0;
}


int main2(int argc, char* argv[])
{
	Mat image1, hsv, h, s, v;
	Mat hC[3], sC[3], vC[3];
	image1 = cv::imread(std::string("images/0000000000.png"), CV_LOAD_IMAGE_COLOR);
	cv::imshow("Color", image1);
	cv::createTrackbar("minH", "Color", 0, 255, updateMinH);
	cv::createTrackbar("maxH", "Color", 0, 255, updateMaxH);
	cv::createTrackbar("minS", "Color", 0, 255, updateMinS);
	cv::createTrackbar("maxS", "Color", 0, 255, updateMaxS);
	cv::createTrackbar("minV", "Color", 0, 255, updateMinV);
	cv::createTrackbar("maxV", "Color", 0, 255, updateMaxV);
	cv::createTrackbar("nb_img", "Color", 0, 836, updateNBImage);
	while (true)
	{
		image1 = cv::imread(getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		cvtColor(image1, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(minH, 0, 0), Scalar(maxH, 255, 255), h);
		inRange(hsv, Scalar(0, minS, 0), Scalar(255, maxS, 255), s);
		inRange(hsv, Scalar(0, 0, minV), Scalar(255, 255, maxV), v);
		split(h, hC);
		split(s, sC);
		split(v, vC);
		sC[0].copyTo(hC[1]);
		vC[0].copyTo(hC[2]);
		merge(hC, 3, h);
		cv::imshow("Color", h);
		cv::imshow("Originale", image1);
		waitKey(1);
	}
	cv::waitKey();
	std::cout << "end" << std::endl;
	return 0;
}


int main(int argc, char* argv[])
{
	main3(argc, argv);
}