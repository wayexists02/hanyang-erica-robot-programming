#ifndef __EDRONE_ADAPTOR_HPP__
#define __EDRONE_ADAPTOR_HPP__

#include <ros/ros.h>
#include <string>
#include <jsoncpp/json/json.h>
#include <actionlib/server/simple_action_server.h>
#include "drone_message/DroneCommandAction.h"

enum class FLAGS {
    INFO, CMD
};

class EdroneAdaptor
{
public:
    EdroneAdaptor(ros::NodeHandle* _nh);
    virtual ~EdroneAdaptor();

    void createAdaptor();
    std::string getDataFromDrone();
    void forward(std::string& data);
    // void handleCmd(const drone_message::DroneCommand::ConstPtr& msg_ptr);
    void handleCmdAction(const drone_message::DroneCommandGoalConstPtr& act_ptr);
    void startActionServer();
    void sendCmd();
    bool getFeedback();
    void test();

protected:
    ros::NodeHandle* nh;

    ros::Publisher info_pub;
    ros::Subscriber cmd_sub;
    actionlib::SimpleActionServer<drone_message::DroneCommandAction> cmd_act_serv;

    Json::Value root;
    Json::StyledWriter writer;
    Json::Reader reader;

    // std::mutex mutex;

    int pipefd_in[2];
    int pipefd_out[2];
    int pid;

    FLAGS flag;

    int in_fd;
    int out_fd;
};

#endif