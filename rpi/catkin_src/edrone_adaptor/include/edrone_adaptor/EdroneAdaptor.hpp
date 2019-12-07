#ifndef __EDRONE_ADAPTOR_HPP__
#define __EDRONE_ADAPTOR_HPP__

#include <ros/ros.h>
#include <string>
#include <std_msgs/Bool.h>
#include <drone_message/DroneCommand.h>
#include <jsoncpp/json/json.h>

enum class FLAGS {
    INFO, CMD
};

class EdroneAdaptor
{
public:
    EdroneAdaptor(ros::NodeHandle* _nh);
    virtual ~EdroneAdaptor();

    void createAdaptor();
    // std::string getDataFromDrone();
    // void forward(std::string& data);
    void handleCmd(const drone_message::DroneCommand::ConstPtr& msg_ptr);
    void sendCmd();
    void test();

    void resetMessage();

    void handleStop(const std_msgs::Bool::ConstPtr& msg_ptr);

protected:
    ros::NodeHandle* nh;

    ros::Publisher info_pub;
    ros::Subscriber cmd_sub;
    ros::Subscriber stop_sub;

    Json::Value root;
    Json::StyledWriter writer;
    Json::Reader reader;

    int pipefd_in[2];
    int pipefd_out[2];
    int pid;

    FLAGS flag;

    bool stop;

    int in_fd;
    int out_fd;
};

#endif