#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
bool manual = false;
int minH, maxH = 255, minS, maxS = 255, minV, maxV = 255, nb_img = 0, i;
vector<Point> triangle;
Point2f cCenter;
Point2f points[4];
float cRadius, area, size;
vector<vector<Point>> contours;
RotatedRect rect;

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
	manual = true;
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

void getContoursAndDraw(const Mat* baseImage, const Scalar* rectColor, const Scalar* circleColor, const Scalar* triangleColor, Mat* imageToDrawOn)
{

	findContours(*baseImage, contours, RETR_LIST, CV_CHAIN_APPROX_NONE);
	int i = 0;
	for (const vector<Point>& cnt : contours)
	{
		area = contourArea(cnt);
		if (area > 400 && area < 30000)
		{
			size = minEnclosingTriangle(cnt, triangle);
			if (area > size * 0.9)
			{
				line(*imageToDrawOn, triangle[0], triangle[1], *triangleColor);
				line(*imageToDrawOn, triangle[1], triangle[2], *triangleColor);
				line(*imageToDrawOn, triangle[2], triangle[0], *triangleColor);
			}
			else
			{
				minEnclosingCircle(cnt, cCenter, cRadius);
				if (area > 3.14 * cRadius * cRadius * 0.75)
				{
					circle(*imageToDrawOn, cCenter, cRadius, *circleColor);
				}
				else
				{
					rect = minAreaRect(cnt);
					rect.points(points);
					if (rect.size.height * rect.size.width * 0.7 < area)
					{
						line(*imageToDrawOn, points[0], points[1], *rectColor);
						line(*imageToDrawOn, points[1], points[2], *rectColor);
						line(*imageToDrawOn, points[2], points[3], *rectColor);
						line(*imageToDrawOn, points[3], points[0], *rectColor);
					}
				}
			}
		}
		i++;
	}
}



int main3(int argc, char* argv[])
{
	Scalar green(0, 255, 0);
	Scalar red(0, 0, 255);
	Scalar blue(255, 0, 0);
	Mat image1, hsv, tmp, tmp2;
	Scalar colors[] = {
		Scalar(0, 147, 30), Scalar(5, 230, 50),
		Scalar(0, 0, 78), Scalar(25, 107, 85),
		Scalar(12, 172, 35), Scalar(21, 200, 61),
		Scalar(92, 0, 175), Scalar(100, 255, 255),
		Scalar(94, 0, 74), Scalar(109, 109, 255),
		Scalar(94, 63, 59), Scalar(109, 100, 66),
		Scalar(96, 65, 67), Scalar(106, 108, 89),
		Scalar(101, 75, 59), Scalar(111, 100, 69),
		Scalar(105, 181, 50), Scalar(115, 245, 214),
		Scalar(104, 131, 0), Scalar(115, 255, 50),
		Scalar(107, 89, 32), Scalar(110, 156, 44),
		Scalar(108, 198, 171), Scalar(117, 239, 231),
		Scalar(160, 86, 55), Scalar(184, 195, 93),
		Scalar(120, 27, 44), Scalar(125, 31, 48),
		Scalar(158, 88, 55), Scalar(184, 202, 93),
		Scalar(163, 158, 0), Scalar(180, 205, 28),
		Scalar(171, 82, 0), Scalar(197, 255, 47)
	};
	image1 = cv::imread(std::string("images/0000000000.png"), CV_LOAD_IMAGE_COLOR);
	cv::imshow("Color", image1);
	cv::createTrackbar("nb_img", "Color", 0, 836, updateNBImage);
	while (nb_img < 837)
	{
		image1 = cv::imread(getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		cvtColor(image1, hsv, CV_BGR2HSV);
		inRange(hsv, colors[0], colors[1], tmp);
		for (i = 2; i < 34; i+=2)
		{
			inRange(hsv, colors[i], colors[i+1], tmp2);
			max(tmp, tmp2, tmp);
		}
		getContoursAndDraw(&tmp, &red, &green, &green, &image1);
		blur(tmp, tmp, Size(3, 3));
		getContoursAndDraw(&tmp, &red, &green, &green, &image1);
		cv::imshow("Color", image1);
		waitKey(1);
		if (!manual)
			nb_img++;
	}
	cv::waitKey();
	std::cout << "end" << std::endl;
	return 0;
}


int main2(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "Pleas provide the image's folder's path" << std::endl;
		return -1;
	}
	Scalar green(0, 255, 0);
	Scalar red(0, 0, 255);
	Mat image1, hsv, h, s, v, tmp;
	Mat hC[3], sC[3], vC[3];
	image1 = cv::imread(std::string(argv[2]) + std::string("images/0000000000.png"), CV_LOAD_IMAGE_COLOR);
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
		vector<vector<Point>> contours;
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
		inRange(hsv, Scalar(minH, minS, minV), Scalar(maxH, maxS, maxV), tmp);
		blur(tmp, tmp, Size(5, 5));
		findContours(tmp, contours, RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS);
		int i = 0;
		for (const vector<Point>& cnt : contours)
		{
			if (isContourConvex(cnt))
				drawContours(image1, contours, i, green);
			else
				drawContours(image1, contours, i, red);
			i++;
		}
		cv::imshow("Originale", image1);
		waitKey(1);
	}
	cv::waitKey();
	std::cout << "end" << std::endl;
	return 0;
}


int main(int argc, char* argv[])
{
	if (argc > 1)
		if (strcmp(argv[1], "colors") == 0)
			return main2(argc, argv);
	return main3(argc, argv);
}