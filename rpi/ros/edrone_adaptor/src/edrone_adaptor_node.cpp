#include <ros/ros.h>
#include "edrone_adaptor/EdroneAdaptor.hpp"


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "edrone_adaptor_node");
    ros::NodeHandle nh;

    EdroneAdaptor drone_adaptor(&nh);
    drone_adaptor.createAdaptor();

    ros::Rate rate(10);

    while (ros::ok()) {
        drone_adaptor.test();
        rate.sleep();
    }

    return 0;
}
