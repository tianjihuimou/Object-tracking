// CAMSHIFT TRACKER

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;
Point originPoint;
Rect selectedRect;
bool selectRegion = false;
int trackingFlag = 0;

const char* keys =
{
	"{help h usage ? | | print this message}"
    "{@video | | Video file, if not defined try to use webcamera}"
};


// 鼠标选定区域。
void onMouse(int event, int x, int y, int, void*)
{
    if(selectRegion)
    {
        selectedRect.x = MIN(x, originPoint.x);
        selectedRect.y = MIN(y, originPoint.y);
        selectedRect.width = std::abs(x - originPoint.x);
        selectedRect.height = std::abs(y - originPoint.y);
        //获取选定区域。
        selectedRect &= Rect(0, 0, image.cols, image.rows);
    }
    
    switch(event)
    {
        case CV_EVENT_LBUTTONDOWN:
            originPoint = Point(x,y);
            selectedRect = Rect(x,y,0,0);
            selectRegion = true;
            break;
            
        case CV_EVENT_LBUTTONUP:
            selectRegion = false;
            if( selectedRect.width > 0 && selectedRect.height > 0 )
            {
                trackingFlag = -1;
            }
            break;
    }
}

int main(int argc, char* argv[])
{
    // Create the capture object
    // 0 -> input arg that specifies it should take the input from the webcam
    //VideoCapture cap(0);
    CommandLineParser parser(argc, argv, keys);
    //If requires help show
    if (parser.has("help"))
	{
	    parser.printMessage();
	    return 0;
	}

	String videoFile= parser.get<String>(0);
	
	// Check if params are correctly parsed in his variables
	if (!parser.check())
	{
	    parser.printErrors();
	    return 0;
	}

	VideoCapture cap; // open the default camera
	if(videoFile != "")
		cap.open(videoFile);
	else
		cap.open(0);
    char ch;
    Rect trackingRect;
    
    // range of values for the 'H' channel in HSV ('H' stands for Hue)
    float hueRanges[] = {0,180};
    const float* histRanges = hueRanges;
    
    // min value for the 'S' channel in HSV ('S' stands for Saturation)
    int minSaturation = 40;
    
    // min and max values for the 'V' channel in HSV ('V' stands for Value)
    int minValue = 20, maxValue = 245;
    
    // size of the histogram bin
    int histSize = 8;
    
    string windowName = "CAMShift Tracker";
    namedWindow(windowName, 0);
    setMouseCallback(windowName, onMouse, 0);
    
    Mat frame, hsvImage, hueImage, mask, hist, backproj;
    
    // Image size scaling factor for the input frames from the webcam
    float scalingFactor = 0.75;
    
    // Iterate until the user presses the Esc key
    while(true)
    {
    
        cap >> frame;
    
        if(frame.empty())
            break;
        
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);
    
     
        frame.copyTo(image);
    
        // Convert to HSV colorspace
        cvtColor(image, hsvImage, COLOR_BGR2HSV);
        
        if(trackingFlag)
        {
           
            // 把指定范围内的结果放在mask 中。
            inRange(hsvImage, Scalar(0, minSaturation, minValue), Scalar(180, 256, maxValue), mask);
            
            // Mix the specified channels
            int channels[] = {0, 0};
            hueImage.create(hsvImage.size(), hsvImage.depth());
            mixChannels(&hsvImage, 1, &hueImage, 1, channels, 1);
            
            if(trackingFlag < 0)
            {
                // 创建基于选定区域的图像。
                Mat roi(hueImage, selectedRect), maskroi(mask, selectedRect);
                
                // 创建直方图并将其正常化。
                calcHist(&roi, 1, 0, maskroi, hist, 1, &histSize, &histRanges);
                normalize(hist, hist, 0, 255, CV_MINMAX);
                
                trackingRect = selectedRect;
                trackingFlag = 1;
            }
            
            // 计算直方图反向投影。
            calcBackProject(&hueImage, 1, 0, hist, backproj, &histRanges);
            backproj &= mask;
            RotatedRect rotatedTrackingRect = CamShift(backproj, trackingRect, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
            
            // 检查选定区域面积是否过小。
            if(trackingRect.area() <= 1)
            {
                // 使用偏移的值确保最小尺寸。
                int cols = backproj.cols, rows = backproj.rows;
                int offset = MIN(rows, cols) + 1;
                trackingRect = Rect(trackingRect.x - offset, trackingRect.y - offset, trackingRect.x + offset, trackingRect.y + offset) & Rect(0, 0, cols, rows);
            }
            
            // 绘制椭圆。
            ellipse(image, rotatedTrackingRect, Scalar(0,255,0), 3, CV_AA);
        }
        
        // 使用兴趣区域的负面影响。
        if(selectRegion && selectedRect.width > 0 && selectedRect.height > 0)
        {
            Mat roi(image, selectedRect);
            bitwise_not(roi, roi);
        }
        
        // 显示输出图像。
        imshow(windowName, image);
        
        // Get the keyboard input and check if it's 'Esc'
        // 27 -> ASCII value of 'Esc' key
        ch = waitKey(30);
        if (ch == 27) {
            break;
        }
    }
    
    return 0;
}
