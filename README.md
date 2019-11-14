# Robot Programming

김영평, 송희원, 이재영


ROS(Robot Operating Systems)를 공부하는 수업으로, 이번 프로젝트에서는 ROBOLINK사의 Codrone이라는 드론 위에 ROS와 그 노드들을 프로그래밍하고 구동해봄으로써, ROS를 학습하고자 한다.



## CoDrone

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





## Plan

- [X] 작동 확인
- [X] ROS를 raspberry pi zero 보드에 설치
- [X] 서버와 raspberry pi zero 보드간에 ROS 통신
- [X] 카메라 영상을 받아와서 확인
- [x] PC에서 드론 제어

- [ ] 손 모양 데이터를 수집할 것



### Consideration

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
