#include "ExtractFeat.h"

using namespace cv;
using namespace std;

void undistortImg(Mat &img)
{
	double camMatrixData[] = { 2898.4947, 0, 1006.1504, 0, 2898.7942, 621.3726, 0, 0, 1 };
	double distCoeffsData[] = { -0.2296, -0.5837, 0, 0 };

	Mat camMatrix = Mat(3, 3, CV_64F, camMatrixData);
	Mat distCoeffs = Mat(1, 4, CV_64F, distCoeffsData);
	
	Mat undistorted;
	undistort(img, undistorted, camMatrix, distCoeffs);
	img = undistorted;
}

void loadImages(const String &path, vector<Mat> &images)
{
	vector<String> fn;
	glob(path, fn, true);
	for (int i = 0; i < fn.size(); i++) 
	{
		Mat im = imread(fn[i]);
		if (im.empty()) 
		{
			continue;
		}
		
		undistortImg(im);

		int x = 0, y = 330;
		int width = 1936, height = 1037;
		im = im(Rect(x, y, width - x, height - y));

		images.push_back(im);
	}
}

void testing()
{
	vector<Mat> images;
	loadImages("../data/images/testing/*.tif", images);
	ExtractFeat classifier;
	classifier.testingMode = true;
	classifier.runTesting(images);
}

void training()
{
	vector<Mat> images;
	loadImages("../data/images/training/*.tif", images);
	ExtractFeat classifier;
	classifier.testingMode = false;
	classifier.runTraining(images);
}

int main()
{
	//training();
	
	testing();

	return 0;
}
