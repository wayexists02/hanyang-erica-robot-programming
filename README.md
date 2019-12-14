# Robot Programming

**팀 구성**: 김영평, 송희원, 이재영


ROS(Robot Operating Systems)를 공부하는 수업으로, 이번 프로젝트에서는 ROBOLINK사의 Codrone이라는 드론 위에 ROS와 그 노드들을 프로그래밍하고 구동해봄으로써, ROS를 학습하고자 한다.

## Goals
#### 기본 동작
- [X] 작동 확인
- [X] ROS를 raspberry pi zero 보드에 설치
- [X] 서버와 raspberry pi zero 보드간에 ROS 통신

#### 손 인식
- [X] 카메라 영상을 받아와서 확인
- [X] 손 모양 데이터를 수집할 것
- [X] 수신호로 드론에게 명령을 내리기

#### 링 통과
- [X] 드론을 코드로 제어하기 (ROS topic으로 드론 제어 성공)
- [X] 색깔이 있는 링 인식
- [ ] 색깔이 있는 링 통과하기

## Preparation

1. server/catkin_src/* 의 모든 파일을 서버의 catkin_ws/src/ 에 넣는다.
2. rpi/ros/codrone_alpha 를 raspberry pi 의 $HOME/ 에 넣는다.
3. rpi/ros/catkin_src/* 의 모든 파일을 raspberry pi 의 catkin_ws/src/ 에 넣는다.
4. 각각 빌드한다.

## Execution

1. 서버와 코드론을 통신 가능한 네트워크로 연결하고, 서버에 ```roscore```를 실행
2. 서버에서 ```roslaunch codrone_alpha launch.launch```
3. Raspberry pi에서 ```roslaunch codrone_alpha_pi launch.launch```를 



## Detail

먼저, 라즈베리파이에서 이미지를 PC로 송신하는 노드가 존재한다. 이 노드로부터 서버가 이미지를 받고 다음과 같은 연산을 거친다.

![image](https://user-images.githubusercontent.com/26874750/70841270-719a8d00-1e5c-11ea-835e-c59f576dac91.png)

작동 구조는 현재까지 위와 같다. 전반적으로 stop 토픽이 드론의 비상 정지를 준비하면서, 어느 토픽보다 가장 높은 우선순위로 동작하게 된다. 그 다음으로, mission 서비스인데, mission 서비스가 ring detection mode, sign detection mode를 스위칭하는 멀티플랙서 역할을 수행한다. Ring detection mode에서는 전방에 링을 인식해서 그 링을 통과하도록 드론이 자동 조종되고 hand detection mode에서는 드론을 손 모양으로 조종이 가능하다. 이 두 모드가 동시에 조종명령을 보내면 예상치못한 상황이 너무 많으므로, 멀티플랙서 구조를 채택했다.

두 모드에서 각각 다른 ROS node가 동작하며, 이 노드들은 라즈베리의 edrone adaptor 노드로 명령을 토픽으로 전송한다. Edrone adaptor는 말 그대로 edrone 라이브러리(python3)와 ROS 노드(C++, python2)를 연결해주는 어댑터 역할을 하며, edrone adaptor는 fork()를 이용해 자식프로세스를 생성하고 python3을 구동하고 edrone library를 호출하도록 되어 있다. 그리고, edrone adaptor와 python3는 pipe를 통해 통신하게 된다.

이런 구조를 채택한 이유는, ROS가 공식적으로 python2만 지원하며, edrone library는 python3만 지원하기 때문이다. 따라서, 라이브러리를 직접 수정하던지, 다른 방법이 필요했고, 우리는 하위 프로세스를 python3으로 생성하는 방법을 생각했다.

Edrone adaptor가 하는 역할 중 하나는 드론 명령에 해당하는 ROS topic(DroneCommand.msg)을 json 형태로 변환시켜서 python3 프로세스에게 전달한다. Python3 프로세스는 edrone library를 직접 이용하면서 json을 해석하고 edrone library를 통해 직접 명령을 내린다.



# About CoDrone

참조: https://www.robolink.com/codrone/



## 인터페이스

그냥 파이썬으로 하자. ```sudo pip3 install e-drone```으로 e-drone 설치. python2버전은 e-drone라이브러리가 지원을 안한다.



### Connect

E-Drone 라이브러리를 이용한다.

http://dev.byrobot.co.kr/documents/kr/products/e_drone/library/python/e_drone/

먼저, 라즈베리 zero 보드에서 블루투스를 비활성화한다.

- ```/boot/config.txt```에서 다음 라인 추가

  ```
  enable_uart=1
  dtoverlay=pi3-disable-bt
  ```

- ```sudo systemctl disable hciuart```

- ```sudo raspi-config```에서 ```networking option```의 ```serial```을 선택한다. 그리고, 첫 번째로 뜨는 serial login 어쩌구에서는 No를, 그 다음에 뜨는 serial connection 관련해서는 yes를 클릭한다.(serial login에서 yes하면 두번째 창은 뜨지 않는다.)

- Baudrate를 확인하고 변경한다.

  ```stty -F /dev/ttyAMA0 115200```

- 재부팅

그리고, ```sudo chmod```로 ```/dev/ttyAMA0```의 퍼미션을 수정하던, 슈퍼유저 권한으로 들어간다.

(슈퍼유저 들어가는 법은 ```su -```또는 ```sudo bash```중 하나 bash 창에서 입력)

다음은 연결하는 파이썬 코드.

```python
from e_drone.drone import *
from e_drone.protocol import *

drone = Drone()
drone.open()

# ...무슨 일이던 수행한다.

drone.close()
```





### Basic Control

```python
from e_drone.drone import *
from e_drone.protocol import *
from time import sleep

# 드론 객체 생성
drone = Drone()
# 드론 연결 (마지막으로 연결된 시리얼 포트로 연결함)
drone.open()

print("Takeoff")
drone.sendTakeOff()
sleep(0.01)

# 5초간 이륙할 시간을 준다.
sleep(5)

print("Hovering")
for i in range(5):
    drone.sendControlWhile(0, 0, 0, 0, 1000)
    sleep(1)
    print(i)
    
print("Throttle down")
for i in range(5):
    drone.sendControlWhile(0, 0, 0, -10, 1000)
    sleep(1)
    print(i)
    
print("Landing")
drone.sendLanding()
sleep(0.01)

print("Stop")
drone.sendStop()
sleep(0.01)

drone.close()
```



### LED 켜기

```python
from e_drone.drone import *
from e_drone.protocol import *
from time import sleep

# 드론 객체 생성
drone = Drone()
# 드론 연결
drone.open()

# 파랑색으로
drone.sendLightManual(DeviceType.Drone, LightFlagsDrone.BodyBlue.value, 100)
sleep(3)

# 초록색으로
drone.sendLightManual(DeviceType.Drone, LightFlagsDrone.BodyGreen.value, 100)
sleep(3)

drone.close()
```





### Information Retrieve

드론에 있는 Request를 보내는 메소드를 호출하면 정보를 얻어올 수 있다.





## Project Ideas

- 바닥 라인을 따라가면서 링을 인식하고 통과하면서 주행하기.
- 객체를 인식하고 사람을 찾아서 따라다니기
- 손 모양을 인식해서 드론의 행동을 제어하기



## Consideration

ROS 통신 방법 (Topic, Service, [Action *- optional*])을 완전히 이해할 것

서로 보낼 topic 이름과 service 이름을 정의하고 팀원에게 공개할 것





## 학습 자료

오로카 카페: https://cafe.naver.com/openrt/2360

Codrone Arduino API: https://www.robolink.com/codrone-functions-guide/

Codrone Python API: http://docs.robolink.com/

Codrone Github: https://github.com/RobolinkInc/CoDrone

http://dev.byrobot.co.kr/documents/kr/products/e_drone/library/python/e_drone/02_system/

eDrone API: http://dev.byrobot.co.kr/documents/kr/products/e_drone/library/python/e_drone/

CoDrone II 한국어 페이지: http://www.robolink.co.kr/web/cate02/product01.mn.php
