
#include <signal.h>
#include <unistd.h>
#include <ros/ros.h>
#include "edrone_adaptor/EdroneAdaptor.hpp"


EdroneAdaptor* drone_adaptor;


void int_handler(int signo);

int main(int argc, char* argv[])
{
    struct sigaction sa = {0,};
    sa.sa_handler = int_handler;

    sigaction(SIGINT, &sa, NULL);

    // 노드 생성
    ros::init(argc, argv, "edrone_adaptor_node");
    ros::NodeHandle nh;

    // 드론 어댑터 객체 생성 후 초기화
    drone_adaptor = new EdroneAdaptor(&nh);
    drone_adaptor->createAdaptor();
    drone_adaptor->startActionServer();

    // 1 FPS 로 실행
    ros::Rate rate(1);

    std::string data;

    while (ros::ok()) {
        // TEST
        // drone_adaptor.test();

        // // 데이터를 드론으로부터 받아옴
        // data = drone_adaptor->getDataFromDrone();
        // if (data == "") continue;
        
        // // 데이터를 ROS 토픽으로 포워딩
        // drone_adaptor->forward(data);

        // drone_adaptor->sendCmd();

        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}

void int_handler(int signo)
{
    delete drone_adaptor;
    ros::shutdown();
}
