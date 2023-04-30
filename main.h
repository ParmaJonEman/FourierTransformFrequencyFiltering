//
// Created by jonem on 2/19/2023.
//

#ifndef FREQFILTER_MAIN_H
#define FREQFILTER_MAIN_H
#include <conio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <locale.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <iterator>
using namespace cv;
using namespace std;
void mouseCallback(int event, int x, int y, int d, void *userdata);
void initCallback(string windowName, Mat *frequencyMask);
Mat createMaskForFilter(Mat magnitudeImage);
void swapMat(Mat *a, Mat *b);
void swapQuadrants(Mat *src);
Mat applyMagnitudeMask(Mat *complexImage, Mat magnitudeMask);
Mat extractAndTransformMagnitude(Mat *complexImage);
Mat createPaddedImage(Mat originalImage);
Mat filterFrequency(Mat image);
static int parseParameters(int argc, char** argv, String* imageFile1, String* imageFile2);

#endif //FREQFILTER_MAIN_H
