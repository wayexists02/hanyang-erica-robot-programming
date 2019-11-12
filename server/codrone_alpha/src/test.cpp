#include <ros/ros.h>
#include "codrone_alpha/Test.hpp"


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh;

    Test test(nh);

    ros::Rate rate(10);

    while (ros::ok()) {
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}