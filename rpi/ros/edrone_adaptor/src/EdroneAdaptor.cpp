#include <ros/ros.h>
#include <std_msgs/String.h>

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstring>

#include "edrone_adaptor/EdroneAdaptor.hpp"


/**
 * 생성자
 */
EdroneAdaptor::EdroneAdaptor(ros::NodeHandle* _nh)
    : nh(_nh)
{
    info_pub = nh->advertise<std_msgs::String>("codrone_info", 0);
}

EdroneAdaptor::~EdroneAdaptor()
{
    // 이 객체가 파괴될 때, 하위 프로세스(e-drone 어댑터)를 종료시키고 자원 회수
    // 그리고 파이프를 닫음
    close(this->in_fd);
    close(this->out_fd);
    kill(this->pid, SIGINT);
    waitpid(this->pid, NULL, 0);
}

/**
 * 이 객체를 초기화하는 메소드
 * e-drone을 위한 어댑터 프로세스를 python3으로 생성 및 실행
 */
void EdroneAdaptor::createAdaptor()
{
    // python3 과 통신하기 위한 파이프 생성
    if (pipe(this->pipefd) < 0) {
        // 파이프가 생성실패하면 에러 메시지 뱉고 종료
        ROS_ERROR("CANNOT open pipe!");
        ros::requestShutdown();
        return;
    }
    
    // 하위 프로세스 생성
    this->pid = fork();

    if (this->pid < 0) {
        // 프로세스 생성 실패일때
        ROS_ERROR("CANNOT create child process!");
        ros::requestShutdown();
        return;
    }
    else if (this->pid == 0) {
        // 하위 프로세스라면, 이것을 실행함
        char fd1[16] = {0};
        char fd2[16] = {0};

        // 파이프 file descriptor를 하위 프로세스에게 전달해줘야함
        // 그러기 위해 file descriptor들을 문자열로 바꿈
        sprintf(fd1, "%d", this->pipefd[0]);
        sprintf(fd2, "%d", this->pipefd[1]);

        // 현재 사용자의 홈 디렉토리 경로를 얻어옴
        char* homedir = getenv("HOME");

        // 실행파일의 경로를 설정. 반드시 $HOME/codrone_alpha/ 아래에 있을 것
        char exec_file[64] = {0,};
        sprintf(exec_file, "%s/codrone_alpha/main.py", homedir);

        // python3 으로 e-drone 어뎁터 생성
        execlp("/usr/bin/python3", "/usr/bin/python3", exec_file, fd1, fd2, NULL);
    }
    else {
        // 부모 프로세스라면, 이것을 실행함
        this->in_fd = this->pipefd[0];
        this->out_fd = this->pipefd[1];
        ROS_INFO("Child process was started!");
        ROS_INFO("IN FD: %d", this->in_fd);
        ROS_INFO("OUT FD: %d", this->out_fd);
    }
}

/**
 * 테스트용
 */
void EdroneAdaptor::test()
{
    write(this->out_fd, "TEST", strlen("TEST"));
}

/**
 * 드론으로부터 데이터를 받아옴
 * 
 * Returns:
 * @data        json 문자열
 */
std::string EdroneAdaptor::getDataFromDrone()
{
    // 일단 json 데이터의 길이를 받아옴
    char len_of_data[8] = {0,};
    read(this->in_fd, len_of_data, 8);

    // 데이터를 받아옴
    char buf[256] = {0,};
    read(this->in_fd, buf, atoi(len_of_data));

    std::string data(buf);

    return data;
}

/**
 * 데이터를 메시지로 만들어서 그대로 포워딩
 * 
 * Parameters
 * @data    데이터 문자열(json 형태)
 */
void EdroneAdaptor::forward(std::string& data)
{
    // 메시지 객체 1번만 생성 후 재활용
    static std_msgs::String msg;

    // 메시지 객체에 데이터 채우기
    msg.data = data;

    // 메시지 퍼블리싱
    info_pub.publish(msg);

    // 메시지 데이터 클리어
    msg.data.clear();
}
