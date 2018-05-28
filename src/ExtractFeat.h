#pragma once

#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>


using namespace cv;
using namespace std;

struct Fillet 
{
	String name;
	double hist_mean[2] = { 0 };
	double area = 0;
	double convexity = 0;
	double skinArea = 0;
	double largestNotch = 0;
	bool bloodstain = false;
	int classification = 0;
	string reason = "None";

	Rect boundRect;
	vector<Point> contour;
	Mat img, bin;
};

class ExtractFeat
{
public:
	bool testingMode = false;
	String data_file_path = "../data/features.dat";
	String data_file_path_classification = "../data/classification.dat";

	void clearFileContent();
	void displayImg(const String &name, const Mat &img);
	void makeBinary(const Mat &img, Mat &bin);

	void Classify(const Fillet &fillet);
	
	void getMean(Fillet &fillet);
	void getBloodstains(Fillet &fillet);
	void getNotches(Fillet &fillet);
	void getShape(Fillet &fillet);
	void getSkin(Fillet &fillet);

	void saveFeatures(const Fillet &fillet);

	void runTesting(vector<Mat> &images);
	void runTraining(vector<Mat> &images);
};
