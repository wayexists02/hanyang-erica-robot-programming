# Robot Programming



ROS(Robot Operating Systems)를 공부하는 수업으로, 이번 프로젝트에서는 ROBOLINK사의 Codrone이라는 드론 위에 ROS와 그 노드들을 프로그래밍하고 구동해봄으로써, ROS를 학습하고자 한다.



## CoDrone

참조: https://www.robolink.com/codrone/





## 인터페이스

C 또는 C++ 언어로 프로그래밍할 경우, ```#include <CoDrone.h>```를 추가해야 한다. 파이썬의 경우 ```CoDrone``` 패키지를```import```해 준다.

Python의 경우, ```CoDrone```객체를 직접 생성해야 하지만, C++언어의 경우, 객체를 라이브러리에서 생성한다. 따라서 우리는 객체를 직접 생성할 필요가 없으며, ```CoDrone```이라는 이름의 객체의 메소드를 호출하면 된다. 이를 위해서 ```CoDrone.h```파일 제일 아래쪽에 ```extern CoDroneClass CoDrone;```이 선언되어 있고, 어딘가 있는 객체를 코드에서 이용할 수 있다.



### Connect

연결 및 통신하려면, 블루투스로 연결한 후 다음을 시행

```C++
// 드론과 시리얼 통신 시작 band rate는 반드시 115200
CoDrone.begin(115200);

// 이렇게 가까이 있는 드론 연결
CoDrone.AutoConnect(NearbyDrone);
```

파이썬으로도 연결 및 프로그래밍이 가능하다.

```python
import CoDrone

drone = CoDrone.CoDrone() # CoDrone 객체 생성
drone.pair() # 또는 인자로 드론의 unique id 번호 4자리를 넣어준다.
```



### Basic Control

```"CoDrone.h"```에 정의되어 있는 THROTTLE, PITCH, ROLL, YAW를 변경한 후 ```CoDrone.Control()```호출

```C++
THROTTLE = 100;
CoDrone.Control();
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
- [ ] PC에서 드론 제어



- [ ] Arduino API를 이용해서 PC상의 C++로 드론을 제어할 수 있는지 테스트 해볼것
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
