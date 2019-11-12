#ifndef __TEST_HPP__
#define __TEST_HPP__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

class Test
{
public:
    Test(ros::NodeHandle& _nh);
    virtual ~Test();

    void test(const sensor_msgs::Image::ConstPtr& msg);

protected:
    ros::NodeHandle nh;
    ros::Subscriber img_sub;
    ros::Publisher img_pub;
};

#endif