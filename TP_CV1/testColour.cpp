#include "testColour.h"

cv::Scalar dominantColour(cv::Mat& img)
{
	int x_sample = img.cols / 2;
	int y_sample = img.rows / 2;
	int width = img.cols*0.1f;
	int height = img.rows*0.1f;

	//creating a rect for the region of interest
	cv::Rect roi(x_sample, y_sample, width, height);

	//creatin img of th roi
	auto ioi = img(roi);

	//computes mean over roi
	return cv::mean(ioi);

}

bool compareColour(const cv::Scalar& col1, const cv::Scalar& col2, float precision)
{
	float minus = 1 - precision;
	float major = 1 + precision;
	if ((col1[0] <= (col2[0] * major)) && (col1[0] >= (col2[0] * minus)))
	{
		if ((col1[1] <= (col2[1] * major)) && (col1[1] >= (col2[1] * minus)))
		{
			if ((col1[2] <= (col2[2] * major)) && (col1[2] >= (col2[2] * minus)))
				return true;
			else
				return false;
		}
		else
			return false;
	}
	else
		return false;
}