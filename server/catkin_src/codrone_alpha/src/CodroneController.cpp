#include "codrone_alpha/CodroneController.hpp"


CodroneController::CodroneController(ros::NodeHandle& _nh)
    : nh(_nh), is_on_mission(false)
{
    this->cmd_sub = this->nh.subscribe("codrone_cmd", 0, &CodroneController::cmdHandler, this);
    this->cmd_sub_hand = this->nh.subscribe("codrone_cmd_hand", 0, &CodroneController::cmdHandlerHand, this);
    this->cmd_sub_mission = this->nh.subscribe("mission", 0, &CodroneController::handleMission, this);
    this->cmd_pub = this->nh.advertise<drone_message::DroneCommand>("cmd", 0);
}

CodroneController::~CodroneController()
{

}

void CodroneController::updateState()
{
    // 뭘 넣을까.
    ROS_INFO("ON MISSION: %u", this->is_on_mission);
}

void CodroneController::publish()
{
    this->cmd_pub.publish(this->cmd);
    ROS_INFO("Publish CMD");
}

void CodroneController::cmdHandler(const drone_message::DroneCommand::ConstPtr& msg_ptr)
{
    if (!this->is_on_mission) return;

    this->cmd.takeOff = msg_ptr->takeOff;
    this->cmd.pitch = msg_ptr->pitch;
    this->cmd.roll = msg_ptr->roll;
    this->cmd.yaw = msg_ptr->yaw;
    this->cmd.throttle = msg_ptr->throttle;
    this->cmd.lightColorR = msg_ptr->lightColorR;
    this->cmd.lightColorG = msg_ptr->lightColorG;
    this->cmd.lightColorB = msg_ptr->lightColorB;
    
    this->publish();
}

void CodroneController::cmdHandlerHand(const drone_message::DroneCommand::ConstPtr& msg_ptr)
{
    if (this->is_on_mission) return;

    this->cmd.takeOff = msg_ptr->takeOff;
    this->cmd.pitch = msg_ptr->pitch;
    this->cmd.roll = msg_ptr->roll;
    this->cmd.yaw = msg_ptr->yaw;
    this->cmd.throttle = msg_ptr->throttle;
    this->cmd.lightColorR = msg_ptr->lightColorR;
    this->cmd.lightColorG = msg_ptr->lightColorG;
    this->cmd.lightColorB = msg_ptr->lightColorB;
    
    this->publish();
}

void CodroneController::handleMission(const std_msgs::Bool::ConstPtr& msg_ptr)
{
    this->is_on_mission = msg_ptr->data;
}
