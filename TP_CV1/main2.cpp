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
Scalar white(255, 255, 255);
Scalar black(0, 0, 0);

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

void getContoursAndMasks(const Mat* baseImage, const Scalar* rectColor, const Scalar* circleColor, const Scalar* triangleColor, std::vector<Mat>* masks)
{
	findContours(*baseImage, contours, RETR_LIST, CV_CHAIN_APPROX_NONE);
	int i = 0;
	for (const vector<Point>& cnt : contours)
	{
		area = contourArea(cnt);
		if (area > 400 && area < 30000)
		{
			size = minEnclosingTriangle(cnt, triangle);
			if (area > size * 0.9 && !triangle.empty())
			{
				/*Mat mask = Mat::zeros((*baseImage).rows, (*baseImage).cols, CV_8UC1);
				polylines(mask, triangle, true, white, CV_FILLED);
				masks->push_back(mask);*/
				std::cout << "BUG Triangle !!! :@ " << std::endl;
			}
			else
			{
				minEnclosingCircle(cnt, cCenter, cRadius);
				if (area > 3.14 * cRadius * cRadius * 0.75)
				{
					Mat mask = Mat::zeros((*baseImage).rows, (*baseImage).cols, CV_8UC1);
					circle(mask, cCenter, cRadius, white, CV_FILLED);
					masks->push_back(mask);
				}
				else
				{
					rect = minAreaRect(cnt);
					rect.points(points);
					if (rect.size.height * rect.size.width * 0.7 < area)
					{
						Mat mask = Mat::zeros((*baseImage).rows, (*baseImage).cols, CV_8UC1);
						rectangle(mask, points[0], points[2], white, CV_FILLED);
						masks->push_back(mask);
					}
				}
			}
		}
		i++;
	}
}

void initKnownDescriptors(string& prefix, Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, vector<Mat>& knownDescriptors)
{
	Mat image, descriptor;
	vector<KeyPoint> keypoints;
	for (int nb_img = 1; nb_img < 9; nb_img++)
	{
		image = cv::imread(prefix + "classified_images/" + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		featureDetector->detect(image, keypoints, noArray());
		descriptorExtractor->compute(image, keypoints, descriptor);
		knownDescriptors.push_back(descriptor);
		drawKeypoints(image, keypoints, image);
		cv::imshow("Example " + std::to_string(nb_img), image);
		waitKey(1);
	}
}

void getDescriptorAndDrawKeypoints(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, vector<Mat>& descriptors, Mat image, vector<Mat> masks)
{
	Mat descriptor;
	vector<KeyPoint> keypoints;
	for each (const Mat& mask in masks)
	{
		featureDetector->detect(image, keypoints, mask);
		descriptorExtractor->compute(image, keypoints, descriptor);
		drawKeypoints(image, keypoints, image);
		descriptors.push_back(descriptor);
	}
}

int main4(int argc, char* argv[])
{
	Mat mask, maskBGR;
	Scalar green(0, 255, 0);
	Scalar red(0, 0, 255);
	Scalar blue(255, 0, 0);
	Mat image1, hsv, h, s, v, tmp, tmp2, imageGray, canny_output, descriptor;
	Mat hC[3], sC[3], vC[3];
	vector<Mat> masks, knownDescriptors, descriptors;
	Ptr<FeatureDetector> featureDetector = FastFeatureDetector::create();
	Ptr<DescriptorExtractor> descriptorExtractor = BRISK::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<KeyPoint> keypoints;
	vector<DMatch> matches;
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
		Scalar(107, 89, 32), Scalar(110, 156, 44),
		Scalar(108, 198, 171), Scalar(117, 239, 231),
		Scalar(160, 86, 55), Scalar(184, 195, 93),
		Scalar(120, 27, 44), Scalar(125, 31, 48),
		Scalar(158, 88, 55), Scalar(184, 202, 93),
		Scalar(163, 158, 0), Scalar(180, 205, 28),
		Scalar(171, 82, 0), Scalar(197, 255, 47),
		Scalar(104, 131, 0), Scalar(115, 255, 50)
	};
	std::string prefix("");
	if (argc > 1)
		prefix = std::string(argv[1]);

	initKnownDescriptors(prefix, featureDetector, descriptorExtractor, knownDescriptors);

	image1 = cv::imread(prefix + std::string("images/0000000000.png"), CV_LOAD_IMAGE_COLOR);
	cv::imshow("Color", image1);
	cv::createTrackbar("minH", "Color", 0, 255, updateMinH);
	cv::createTrackbar("maxH", "Color", 0, 255, updateMaxH);
	cv::createTrackbar("minS", "Color", 0, 255, updateMinS);
	cv::createTrackbar("maxS", "Color", 0, 255, updateMaxS);
	cv::createTrackbar("minV", "Color", 0, 255, updateMinV);
	cv::createTrackbar("maxV", "Color", 0, 255, updateMaxV);
	cv::createTrackbar("nb_img", "Color", 0, 836, updateNBImage);
	while (nb_img < 837)
	{
		keypoints.clear();
		masks.clear();
		descriptors.clear();
		descriptor = Mat();

		image1 = cv::imread(prefix + getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		cvtColor(image1, hsv, CV_BGR2HSV);
		inRange(hsv, colors[0], colors[1], tmp);
		for (i = 2; i < 34; i += 2)
		{
			inRange(hsv, colors[i], colors[i + 1], tmp2);
			max(tmp, tmp2, tmp);
		}

		getContoursAndMasks(&tmp, &red, &green, &green, &masks);
		blur(tmp, tmp, Size(5, 5));
		getContoursAndMasks(&tmp, &red, &green, &green, &masks);

		getDescriptorAndDrawKeypoints(featureDetector, descriptorExtractor, descriptors, image1, masks);


		/***************************** FIND BEST *****************************/
		int best, bestScore = 0;
		cv::imshow("Color", image1);
		if (!descriptors.empty() && !knownDescriptors.empty()) {
			for (int sign_in_image = 0; sign_in_image < descriptors.size(); sign_in_image++)
			{
				bestScore = 0, best = -1;
				for (int sample_id = 0; sample_id < knownDescriptors.size(); sample_id++)
				{
					matcher->match(descriptors[sign_in_image], knownDescriptors[sample_id], matches);
					int score = 0;
					for (int m = 0; m < matches.size(); m++)
					{
						if (matches[m].distance < 75.0)
							score++;
					}
					if (score > bestScore)
					{
						bestScore = score;
						best = sample_id + 1;
					}
				}
				std::cout << "Panneau " << best << " found at " << sign_in_image << " ( Score : " << bestScore << " )" << std::endl;
			}
		}
		/***************************** FIND BEST *****************************/



			/*
		cvtColor(image1, imageGray, CV_BGR2GRAY);
		blur(imageGray, imageGray, Size(5, 5));
		Canny(imageGray, canny_output, 1.0, 2.0, 3);
		findContours(canny_output, contours, RETR_LIST, CV_CHAIN_APPROX_NONE);
		drawContours(image1, contours, -1, green);
		cv::imshow("Color", image1);
		cvtColor(imageGray, image1, CV_GRAY2BGR);
		cv::imshow("Gray", imageGray);*/
		/***

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

		***/


		waitKey(1);
	}
	cv::waitKey();
	std::cout << "end" << std::endl;
	return 0;
}

/*
int main3(int argc, char* argv[])
{
	Mat mask, maskBGR;
	Scalar green(0, 255, 0);
	Scalar red(0, 0, 255);
	Scalar blue(255, 0, 0);
	Mat image1, hsv, tmp, tmp2, tmp3;
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
		Scalar(107, 89, 32), Scalar(110, 156, 44),
		Scalar(108, 198, 171), Scalar(117, 239, 231),
		Scalar(160, 86, 55), Scalar(184, 195, 93),
		Scalar(120, 27, 44), Scalar(125, 31, 48),
		Scalar(158, 88, 55), Scalar(184, 202, 93),
		Scalar(163, 158, 0), Scalar(180, 205, 28),
		Scalar(171, 82, 0), Scalar(197, 255, 47),
		Scalar(104, 131, 0), Scalar(115, 255, 50)
	};
	std::string prefix("");
	if (argc > 1)
		prefix = std::string(argv[1]);
	image1 = cv::imread(prefix + std::string("images/0000000000.png"), CV_LOAD_IMAGE_COLOR);
	cv::imshow("Color", image1);
	cv::createTrackbar("nb_img", "Color", 0, 836, updateNBImage);
	while (nb_img < 837)
	{
		image1 = cv::imread(prefix + getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
		cv::imshow("Originale", image1);
		mask = Mat::zeros(image1.rows, image1.cols, CV_8UC1);
		cvtColor(image1, hsv, CV_BGR2HSV);
		inRange(hsv, colors[0], colors[1], tmp);
		for (i = 2; i < 34; i+=2)
		{
			inRange(hsv, colors[i], colors[i+1], tmp2);
			max(tmp, tmp2, tmp);
		}
		getContoursAndDraw(&tmp, &red, &green, &green, &mask);
		blur(tmp, tmp, Size(3, 3));
		getContoursAndDraw(&tmp, &red, &green, &green, &mask);
		cvtColor(mask, maskBGR, CV_GRAY2BGR);
		image1 = min(image1, maskBGR);
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
		image1 = cv::imread(std::string(argv[2]) + getImagePrefix(nb_img) + std::to_string(nb_img) + std::string(".png"), CV_LOAD_IMAGE_COLOR);
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
*/

int main(int argc, char* argv[])
{
	/*if (argc > 1)
		if (strcmp(argv[1], "colors") == 0)
			return main2(argc, argv);*/
	return main4(argc, argv);
}