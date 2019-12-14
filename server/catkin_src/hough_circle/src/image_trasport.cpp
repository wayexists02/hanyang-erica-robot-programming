#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/String.h>

using namespace cv;
using namespace std;

int wide = 128;
int height = 128;

int red_num = 0;
int x_avg = 0;
int y_avg = 0;
int x_min = 128;
int x_max = 0;
int y_min = 128;
int y_max = 0;
int radius;

cv::Mat img_hsv;
cv::Mat img_hough;
 
void image_circle(int h_min, int h_max, int s_min, int s_max, int v_min, int v_max)
 {
    int x,y;
    red_num = 0;
    x_avg = 0;
    y_avg = 0;
    x_min = 128;
    x_max = 0;
    y_min = 128;
    y_max = 0;

    cv::Mat img_binary(height, wide, CV_8UC3);
    for(x = 0; x < wide; x++)
    {
      for(y = 0; y < height; y++)
      {
        if(((img_hsv.at<cv::Vec3b>(y,x)[0] >= 0 && img_hsv.at<Vec3b>(y,x)[0] <= 20) || (img_hsv.at<cv::Vec3b>(y,x)[0] > h_min && img_hsv.at<Vec3b>(y,x)[0] < h_max)) && img_hsv.at<Vec3b>(y,x)[1] > s_min && img_hsv.at<Vec3b>(y,x)[1] < s_max && img_hsv.at<Vec3b>(y,x)[2] > v_min && img_hsv.at<Vec3b>(y,x)[2] < v_max)
        {
          img_binary.at<Vec3b>(y,x)[0] = 255; img_binary.at<Vec3b>(y,x)[1] = 255; img_binary.at<Vec3b>(y,x)[2] = 255;
          //for none_hough
          /*
          x_avg += x;
          y_avg += y;
          x_min = min(x_min, x);
          x_max = max(x_max, x);
          y_min = min(y_min, y);
          y_max = max(y_max, y);
          red_num++;
          */
        }
        else
        {
          img_binary.at<Vec3b>(y,x)[0] = 0; img_binary.at<Vec3b>(y,x)[1] = 0; img_binary.at<Vec3b>(y,x)[2] = 0;
        }
      }
    }

    cv::Mat img_gray;
    cv::cvtColor(img_binary, img_gray, CV_BGR2GRAY);
    std::vector<Vec3f> circles;

    HoughCircles (img_gray, circles, CV_HOUGH_GRADIENT, 1, 100, 30, 15, 15, 100);

    for( size_t i = 0; i < circles.size(); i++)
    {
    	radius = 0;
    	cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    	x_avg = cvRound(circles[i][0]);
    	y_avg = cvRound(circles[i][1]);
    	radius = cvRound(circles[i][2]);
    	circle( img_hough, center, 3, Scalar(0, 255, 0), -1, 8, 0);
    	circle( img_hough, center, radius, Scalar(0, 0 ,255), 3, 8, 0);
    }
    //for none_hough
    /*
    x_avg /= red_num;
    y_avg /= red_num;
    cv::Point center(x_avg,y_avg);
    radius = (x_max - x_min + y_max + y_min) / 4;

    cv::circle(img_binary, center, radius, (255, 0, 0), 3);
    cout << "center:" << x_avg << ", " << y_avg << "\tradius: " << radius << "\n" << endl;
    */
    cv::imshow("binary", img_binary);
    cv::imshow("hough", img_hough);
 }

 void imageCallback(const sensor_msgs::ImageConstPtr& msg)
 {
    try
   {
     cv_bridge::CvImagePtr ptr_img;
     ptr_img = cv_bridge::toCvCopy(msg, "bgr8");
     cv::Mat cv_img = ptr_img->image;
     img_hough = cv_img;

     cv::cvtColor(cv_img, img_hsv, CV_BGR2HSV);
     cv::resize(img_hsv, img_hsv, cv::Size(wide, height), 0, 0, CV_INTER_LINEAR);
     cv::resize(cv_img, cv_img, cv::Size(wide, height), 0, 0, CV_INTER_LINEAR);

     cout << "H:" << int(img_hsv.at<cv::Vec3b>(60,60)[0]) << "S:" << int(img_hsv.at<cv::Vec3b>(60,60)[1]) << "V:" << int(img_hsv.at<cv::Vec3b>(60,60)[2]) << endl;
     cv_img.at<Vec3b>(64,64)[0] = 255;
     cv_img.at<Vec3b>(64,64)[1] = 255;
     cv_img.at<Vec3b>(64,64)[2] = 255;
     //cv::imshow("hsv", img_hsv);
     cv::imshow("rgb", cv_img);

     image_circle(160, 181, 50, 251, 50, 250);
     //cv::Mat img(cv::Size(128, 128), CV_32FC3, cv_bridge::toCvCopy(msg, "bgr8")->image);
     //cv::imshow("view", cv_bridge::toCvCopy(msg, "bgr8")->image);
     cv::waitKey(10);
   }
   catch (cv_bridge::Exception& e)
   {
     ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
   }
 }
 
 int main(int argc, char **argv)
 {
   ros::init(argc, argv, "image_listener");
   ros::NodeHandle nh;
   cv::namedWindow("view");
   //cv::startWindowThread();
   ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("circle_info",1000);
   image_transport::ImageTransport it(nh);
   image_transport::Subscriber sub = it.subscribe("image_in", 1, imageCallback);
   ros::Rate loop_rate(10);
   int count = 0;
   while (ros::ok())
   {
   	std_msgs::String msg;
   	std::stringstream ss;
   	//ss << "center" << x_avg << "," << y_avg << "radius " << radius;
   	int circle_info[3] = {x_avg, y_avg, radius};
   	ss << x_avg << " " << y_avg << " " << radius;
   	//ss << "sending" << count;
   	msg.data = ss.str();

   	ROS_INFO("%s", msg.data.c_str());

   	chatter_pub.publish(msg);

   	
   	ros::spinOnce();
   	loop_rate.sleep();
   	//++count;
   }
   
   cv::destroyWindow("view");
 }