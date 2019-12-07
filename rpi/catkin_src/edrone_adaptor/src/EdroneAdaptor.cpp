#include <ros/ros.h>
#include <std_msgs/String.h>

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstring>

#include <jsoncpp/json/json.h>

#include "edrone_adaptor/EdroneAdaptor.hpp"


/**
 * 생성자
 */
EdroneAdaptor::EdroneAdaptor(ros::NodeHandle* _nh)
    : nh(_nh), flag(FLAGS::INFO)
{
    // info_pub = nh->advertise<drone_message::DroneInfo>("codrone_info", 0);
    cmd_sub = nh->subscribe("cmd", 1, &EdroneAdaptor::handleCmd, this);

    // Default 값들.
    root["takeOff"] = "false";
    root["roll"] = "0";
    root["pitch"] = "0";
    root["yaw"] = "0";
    root["throttle"] = "0";
    root["lightColorR"] = "100";
    root["lightColorG"] = "100";
    root["lightColorB"] = "100";
    // root["lightIntensity"] = "100";
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
    if (pipe(this->pipefd_in) < 0) {
        // 파이프가 생성실패하면 에러 메시지 뱉고 종료
        ROS_ERROR("CANNOT open pipe (in)!");
        ros::requestShutdown();
        return;
    }
    // python3 과 통신하기 위한 파이프 생성
    if (pipe(this->pipefd_out) < 0) {
        // 파이프가 생성실패하면 에러 메시지 뱉고 종료
        ROS_ERROR("CANNOT open pipe (out)!");
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
        sprintf(fd1, "%d", this->pipefd_out[0]);
        sprintf(fd2, "%d", this->pipefd_in[1]);

        // 현재 사용자의 홈 디렉토리 경로를 얻어옴
        char* homedir = getenv("HOME");

        // 실행파일의 경로를 설정. 반드시 $HOME/codrone_alpha/ 아래에 있을 것
        char exec_file[64] = {0,};
        sprintf(exec_file, "%s/codrone_alpha/main.py", homedir);

        // python3 으로 e-drone 어뎁터 생성
        execlp("/usr/bin/python3", "/usr/bin/python3", exec_file, fd1, fd2, NULL);
        ROS_ERROR("EXEC ERROR!");
        return;
    }
    else {
        // 부모 프로세스라면, 이것을 실행함
        this->in_fd = this->pipefd_in[0];
        this->out_fd = this->pipefd_out[1];

        close(this->pipefd_in[1]);
        close(this->pipefd_out[0]);

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

void EdroneAdaptor::sendCmd()
{
    std::string cmd = this->writer.write(root);

    int len_of_data = strlen(cmd.c_str());

    char lenbuf[16] = {0,};
    sprintf(lenbuf, "%8d", len_of_data);

    // 명령의 길이를 먼저 전송
    write(this->out_fd, lenbuf, 8);

    // 명령을 전송
    write(this->out_fd, cmd.c_str(), len_of_data);
}

/**
 * 명령어를 받으면 호출되는 콜백
 */
void EdroneAdaptor::handleCmd(const drone_message::DroneCommand::ConstPtr& msg_ptr)
{
    root["takeOff"] = msg_ptr->takeOff;
    root["roll"] = msg_ptr->roll;
    root["pitch"] = msg_ptr->pitch;
    root["yaw"] = msg_ptr->yaw;
    root["throttle"] = msg_ptr->throttle;
    root["lightColorR"] = msg_ptr->lightColorR;
    root["lightColorG"] = msg_ptr->lightColorG;
    root["lightColorB"] = msg_ptr->lightColorB;

    // sendCmd();
}

/**
 * 드론으로부터 데이터를 받아옴
 * 
 * Returns:
 * @data        json 문자열
 */
// std::string EdroneAdaptor::getDataFromDrone()
// {
//     if (this->flag == FLAGS::CMD) return std::string("");

//     // 일단 json 데이터의 길이를 받아옴
//     char len_of_data[8] = {0,};
//     read(this->in_fd, len_of_data, 8);

//     // 데이터를 받아옴
//     char buf[256] = {0,};
//     read(this->in_fd, buf, atoi(len_of_data));

//     std::string data(buf);

//     return data;
// }

/**
 * 데이터를 메시지로 만들어서 그대로 포워딩
 * 
 * Parameters
 * @data    데이터 문자열(json 형태)
 */
// void EdroneAdaptor::forward(std::string& data)
// {
//     // 메시지 객체 1번만 생성 후 재활용
//     static drone_message::DroneInfo msg;
//     static Json::Value info_value;

//     // 메시지 객체에 데이터 채우기
//     this->reader.parse(data, info_value);

//     msg.accel.resize(3);
//     msg.accel[0] = info_value["accelX"].asInt();
//     msg.accel[1] = info_value["accelY"].asInt();
//     msg.accel[2] = info_value["accelZ"].asInt();

//     msg.gyro.resize(3);
//     msg.gyro[0] = info_value["gyroRoll"].asInt();
//     msg.gyro[1] = info_value["gyroPitch"].asInt();
//     msg.gyro[2] = info_value["gyroYaw"].asInt();

//     msg.angle.resize(3);
//     msg.angle[0] = info_value["angleRoll"].asInt();
//     msg.angle[1] = info_value["anglePitch"].asInt();
//     msg.angle[2] = info_value["angleYaw"].asInt();

//     // 메시지 퍼블리싱
//     info_pub.publish(msg);

//     // 메시지 데이터 클리어
//     msg.accel.clear();
//     msg.gyro.clear();
//     msg.angle.clear();

//     this->flag = FLAGS::CMD;
// }
