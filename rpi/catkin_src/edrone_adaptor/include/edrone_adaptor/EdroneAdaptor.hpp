#ifndef __EDRONE_ADAPTOR_HPP__
#define __EDRONE_ADAPTOR_HPP__

#include <ros/ros.h>
#include <string>
#include "drone_message/DroneCommand.h"

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
    void handleCmd(const drone_message::DroneCommand::ConstPtr& msg_ptr);
    void test();

protected:
    ros::NodeHandle* nh;

    ros::Publisher info_pub;
    ros::Subscriber cmd_sub;

    int pipefd[2];
    int pid;

    FLAGS flag;

    int in_fd;
    int out_fd;
};

#endif