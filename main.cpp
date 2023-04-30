#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "main.h"



double distance1;
Point a;
int rightMouseDown = 0;
int leftMouseDown = 0;

void mouseCallback(int event, int x, int y, int d, void *userdata) {
    Mat *magnitude = (Mat *) userdata;
    Point b;
    Point center = Point(magnitude->cols / 2, magnitude->rows / 2);

    double distance2;
    double thickness;
    double radius;

    switch (event) {
        case EVENT_MOUSEMOVE:{
            if(rightMouseDown){
                int xStart, yStart, xEnd, yEnd;
                b = Point(x, y);
                xStart = min(a.x, b.x);
                yStart = min(a.y, b.y);
                xEnd = max(a.x, b.x);
                yEnd = max(a.y, b.y);

                Mat clonedImage = magnitude->clone();
                rectangle(clonedImage, Point(xStart, yStart), Point(xEnd, yEnd), Scalar(0), -1, LINE_8);

                imshow("Magnitude Visible Spectrum", clonedImage);
            }
            else if(leftMouseDown){
                b = Point(x, y);
                distance2 = norm(center - b);
                cout << "distance 2: " << distance2 << " Distance 1: " << distance1 << endl;
                thickness = abs(distance1 - distance2);
                radius = min(distance1, distance2);
                cout << "radius: " << radius << endl;
                cout << "thickness: " << thickness << endl;
                Mat clonedImage = magnitude->clone();
                circle(clonedImage, Point2f(magnitude->cols / 2, magnitude->rows / 2),
                       radius + .5 * thickness, Scalar(0), thickness);
                imshow("Magnitude Visible Spectrum", clonedImage);
            }
            break;
        }
        case EVENT_LBUTTONDOWN:
            leftMouseDown = 1;
            a = Point(x, y);
            distance1 = norm(center - a);
            break;
        case EVENT_LBUTTONUP:
            leftMouseDown = 0;
            b = Point(x, y);
            distance2 = norm(center - b);
            cout << "distance 2: " << distance2 << " Distance 1: " << distance1 << endl;
            thickness = abs(distance1 - distance2);
            radius = min(distance1, distance2);
            cout << "radius: " << radius << endl;
            cout << "thickness: " << thickness << endl;
            circle(*magnitude, Point2f(magnitude->cols / 2, magnitude->rows / 2),
                   radius + .5 * thickness, Scalar(0), thickness);
            imshow("Magnitude Visible Spectrum", *magnitude);
            break;
        case EVENT_RBUTTONDOWN:
            rightMouseDown = 1;
            a = Point(x, y);
            cout << a.x << endl;
            break;
        case EVENT_RBUTTONUP:
            rightMouseDown = 0;

            int xStart, yStart, xEnd, yEnd;
            b = Point(x, y);
            xStart = min(a.x, b.x);
            yStart = min(a.y, b.y);
            xEnd = max(a.x, b.x);
            yEnd = max(a.y, b.y);

            rectangle(*magnitude, Point(xStart, yStart), Point(xEnd, yEnd), Scalar(0), -1, LINE_8);
            imshow("Magnitude Visible Spectrum", *magnitude);
            break;
    }
}

void initCallback(string windowName, Mat *frequencyMask) {
    setMouseCallback(windowName, mouseCallback, frequencyMask);
}

Mat createMaskForFilter(Mat magnitudeImage) {
    imshow("Magnitude Visible Spectrum", magnitudeImage);
    initCallback("Magnitude Visible Spectrum", &magnitudeImage);
    waitKey();
    threshold( magnitudeImage, magnitudeImage, 0, 255, THRESH_BINARY );
    magnitudeImage.convertTo(magnitudeImage, CV_8U);
    return magnitudeImage;
}

void swapMat(Mat *a, Mat *b) {
    Mat tmp;
    a->copyTo(tmp);
    b->copyTo(*a);
    tmp.copyTo(*b);
    tmp.release();
}

void swapQuadrants(Mat *src) {
    int c_x = src->cols / 2;
    int c_y = src->rows / 2;

    Mat q0(*src, Rect(0, 0, c_x, c_y));
    Mat q1(*src, Rect(c_x, 0, c_x, c_y));
    Mat q2(*src, Rect(0, c_y, c_x, c_y));
    Mat q3(*src, Rect(c_x, c_y, c_x, c_y));

    swapMat(&q0, &q3);
    swapMat(&q1, &q2);

    q0.release();
    q1.release();
    q2.release();
    q3.release();
}

Mat applyMagnitudeMask(Mat *complexImage, Mat magnitudeMask) {
    Mat planes[2];
    split(*complexImage, planes);

    Mat real;
    planes[0].copyTo(real, magnitudeMask);

    Mat imaginary;
    planes[1].copyTo(imaginary, magnitudeMask);

    Mat planes2[] = {real, imaginary};
    merge(planes2, 2, *complexImage);

    swapQuadrants(complexImage);
    return *complexImage;
}

Mat extractAndTransformMagnitude(Mat *complexImage) {
    swapQuadrants(complexImage);
    Mat planes[2];
    split(*complexImage, planes);

    Mat magnitudeImage;
    magnitude(planes[0], planes[1], magnitudeImage);

    magnitudeImage.convertTo(magnitudeImage, CV_32F, 255);
    log(magnitudeImage, magnitudeImage);
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

    return magnitudeImage;
}

Mat createPaddedImage(Mat originalImage) {
    int m = getOptimalDFTSize(originalImage.rows);
    int n = getOptimalDFTSize(originalImage.cols);

    Mat padded;
    copyMakeBorder(
            originalImage,
            padded,
            0, m - originalImage.rows,
            0, n - originalImage.cols,
            BORDER_CONSTANT,
            Scalar::all(0)
    );

    padded = padded(Rect(0, 0, padded.cols & -2, padded.rows & -2));
    return padded;
}

Mat filterFrequency(Mat image) {
    Mat padded = createPaddedImage(image);
    image.release();

    Mat planes[2];
    Mat complexImage;
    padded.convertTo(planes[0], CV_32F);
    planes[1] = cv::Mat::zeros(padded.size(), CV_32F);
    merge(planes, 2, complexImage);

    dft(complexImage, complexImage);

    Mat magnitudeImage = extractAndTransformMagnitude(&complexImage);

    Mat filterMask = createMaskForFilter(magnitudeImage);

    imshow("Frequency Mask", filterMask);
    waitKey();

    complexImage = applyMagnitudeMask(&complexImage, filterMask);

    idft(complexImage, complexImage);

    split(complexImage, planes);

    Mat normalizedRealImage;
    normalize(planes[0], normalizedRealImage, 0, 1, NORM_MINMAX);
    normalizedRealImage.convertTo(normalizedRealImage, CV_8U, 255);

    return normalizedRealImage;
}

static int parseParameters(int argc, char** argv, String* imageFile1, String* imageFile2)
{
    String keys =
            {
                    "{help h usage ? |                            | print this message   }"
                    "{@imageFile1      || image you want to filter	}"
                    "{@imageFile2      || filename you want to save as	}"
            };

    // Get required parameters. If any are left blank, defaults are set based on the above table
    // If no directory is passed in, or if the user passes in a help param, usage info is printed
    CommandLineParser parser(argc, argv, keys);
    parser.about("porter-duff v1.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return(0);
    }
    if (parser.has("@imageFile1"))
    {
        *imageFile1 = parser.get<String>("@imageFile1");
    }
    else
    {
        parser.printMessage();
        return(0);
    }
    if (parser.has("@imageFile2"))
    {
        *imageFile2 = parser.get<String>("@imageFile2");
    }
    else
    {
        parser.printMessage();
        return(0);
    }

    return(1);
}

int main(int argc, char **argv) {

    cv::String imageFile1;
    cv::String imageFile2;

    if(!parseParameters(argc, argv, &imageFile1, &imageFile2)){
        return 0;
    };

    Mat originalImage = imread(imageFile1, 0);

    imshow("Noisy Image", originalImage);

    Mat result = filterFrequency(originalImage);

    imshow("Filtered", result);
    waitKey();
    destroyAllWindows();
    imwrite(imageFile2, result);
    return 0;
}