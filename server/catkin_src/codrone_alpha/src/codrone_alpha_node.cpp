#include <ros/ros.h>
#include <drone_message/DroneCommand.h>
#include "codrone_alpha/CodroneController.hpp"


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "codrone_alpha_node");
    ros::NodeHandle nh;

    CodroneController controller(nh);

    ros::Rate rate(5);

    while (ros::ok()) {

        controller.updateState();
        controller.publish();

        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
