#include <ros/ros.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstring>
#include "edrone_adaptor/EdroneAdaptor.hpp"


EdroneAdaptor::EdroneAdaptor(ros::NodeHandle* _nh)
    : nh(_nh)
{

}

EdroneAdaptor::~EdroneAdaptor()
{
    kill(this->pid, SIGINT);
    waitpid(this->pid, NULL, 0);
    close(this->in_fd);
    close(this->out_fd);
}

void EdroneAdaptor::createAdaptor()
{
    if (pipe(this->pipefd) < 0) {
        // pipe error
        ROS_ERROR("CANNOT open pipe!");
        ros::requestShutdown();
        return;
    }
    
    this->pid = fork();

    if (this->pid < 0) {
        // fork error
        ROS_ERROR("CANNOT create child process!");
        ros::requestShutdown();
        return;
    }
    else if (this->pid == 0) {
        // child process
        char fd1[16] = {0};
        char fd2[16] = {0};

        sprintf(fd1, "%d", this->pipefd[0]);
        sprintf(fd2, "%d", this->pipefd[1]);

        char* homedir = getenv("HOME");

        char exec_file[64] = {0,};
        sprintf(exec_file, "%s/codrone_alpha/main.py", homedir);

        execlp("/usr/bin/python3", "/usr/bin/python3", exec_file, fd1, fd2, NULL);
    }
    else {
        // parent process
        this->in_fd = this->pipefd[0];
        this->out_fd = this->pipefd[1];
        ROS_INFO("Child process was started!");
        ROS_INFO("IN FD: %d", this->in_fd);
        ROS_INFO("OUT FD: %d", this->out_fd);
    }
}

void EdroneAdaptor::test()
{
    write(this->out_fd, "TEST", strlen("TEST"));
}
