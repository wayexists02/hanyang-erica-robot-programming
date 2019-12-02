#ifndef __CODRONE_CONTROLLER_HPP__
#define __CODRONE_CONTROLLER_HPP__

#include <ros/ros.h>
#include <drone_message/DroneCommand.h>
#include <std_msgs/Bool.h>

class CodroneController
{
public:
    CodroneController(ros::NodeHandle& _nh);
    virtual ~CodroneController();

    void updateState();
    void publish();
    void cmdHandler(const drone_message::DroneCommand::ConstPtr& msg_ptr);
    void cmdHandlerHand(const drone_message::DroneCommand::ConstPtr& msg_ptr);
    void handleMission(const std_msgs::Bool::ConstPtr& msg_ptr);

protected:
    ros::NodeHandle nh;

    ros::Subscriber cmd_sub;
    ros::Subscriber cmd_sub_hand;
    ros::Subscriber cmd_sub_mission;
    ros::Publisher cmd_pub;

    drone_message::DroneCommand cmd;

    bool is_on_mission;
};

#endif