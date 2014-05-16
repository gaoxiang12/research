#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    stringstream ss;
    ss<<"/home/zero/code/dataset/slam/rgb_index/"<<argv[1]<<string(".png");
    Mat image = imread(ss.str(), 0);
    initModule_nonfree();
    Ptr<FeatureDetector> detector = FeatureDetector::create("FAST");
    for (int i=0; i<100; i++)
        for (int j=0; j<100; j++)
            image.at<unsigned char>(i+300, j+200) = 0;
    vector<KeyPoint> kp;
    detector->detect( image, kp );
    Mat image_keypoints;
    drawKeypoints(image, kp, image_keypoints, Scalar::all(-1), 4);
    imshow( "keypoints", image_keypoints );
    waitKey(0);
    return 0;
}
