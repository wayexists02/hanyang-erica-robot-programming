#ifndef __EDRONE_ADAPTOR_HPP__
#define __EDRONE_ADAPTOR_HPP__

#include <ros/ros.h>
#include <string>

class EdroneAdaptor
{
public:
    EdroneAdaptor(ros::NodeHandle* _nh);
    virtual ~EdroneAdaptor();

    void createAdaptor();
    std::string getDataFromDrone();
    void forward(std::string& data);
    void test();

protected:
    ros::NodeHandle* nh;

    ros::Publisher info_pub;

    int pipefd[2];
    int pid;

    int in_fd;
    int out_fd;
};

#endif