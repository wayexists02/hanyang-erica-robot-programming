#ifndef __EDRONE_ADAPTOR_HPP__
#define __EDRONE_ADAPTOR_HPP__

#include <ros/ros.h>

class EdroneAdaptor
{
public:
    EdroneAdaptor(ros::NodeHandle* _nh);
    virtual ~EdroneAdaptor();

    void createAdaptor();
    void test();

protected:
    ros::NodeHandle* nh;

    int pipefd[2];
    int pid;

    int in_fd;
    int out_fd;
};

#endif