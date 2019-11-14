#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "codrone_alpha/Test.hpp"


Test::Test(ros::NodeHandle& _nh)
    : nh(_nh)
{
    img_sub = nh.subscribe("/usb_cam/image_raw", 0, &Test::test, this);
    img_pub = nh.advertise<sensor_msgs::Image>("/image_in", 0);
}

Test::~Test()
{

}

void Test::test(const sensor_msgs::Image::ConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv bridge error: %s", e.what());
    }

    cv::resize(cv_ptr->image, cv_ptr->image, cv::Size(128, 128));
    this->img_pub.publish(cv_ptr->toImageMsg());
}
