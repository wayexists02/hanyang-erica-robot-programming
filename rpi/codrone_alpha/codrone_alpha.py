from e_drone.drone import *
from e_drone.protocol import *
from time import sleep
import json


class CoDroneAlpha():
    '''
    드론의 컨트롤러와 통신하게 될 객체의 클래스
    객체 생성 후, 반드시 init() 을 호출할 것.
    '''

    def __init__(self):
        self.drone = Drone()

        self.motion = None
        self.altitude = None
        self.on_flying = False

        self.lightColorR = 100
        self.lightColorG = 100
        self.lightColorB = 100


    def init(self):
        '''
        드론 객체 준비
        '''

        # 드론 연결
        self.drone.open()

        # 드론에게 정보를 받은 후 실행될 콜백 함수 등록
        # self.drone.setEventHandler(DataType.Motion, self.motion_handler)
        # self.drone.setEventHandler(DataType.Altitude, self.altitude_handler)

    def update_data(self):
        '''
        드론에게 데이터를 요청하고 저장된 정보를 업데이트한다.
        '''

        # 모션 데이터 요청 후 업데이트
        self.drone.sendRequest(DeviceType.Drone, DataType.Motion)
        sleep(0.1)

        # 고도 데이터 요청 후 업데이트
        # self.drone.sendRequest(DeviceType.Drone, DataType.Altitude)
        # sleep(0.1)

    def get_data(self):
        '''
        저장된 정보를 얻기 위한 메소드

        Returns:
        :data           Dictionary 자료구조로 저장된 데이터
        '''

        if self.motion is None:
            return None

        data = {
            # 모션 센서 정보들
            ## 가속도 정보
            "accelX": self.motion.accelX,
            "accelY": self.motion.accelY,
            "accelZ": self.motion.accelZ,

            ## 자이로 센서 정보
            "gyroRoll": self.motion.gyroRoll,
            "gyroPitch": self.motion.gyroPitch,
            "gyroYaw": self.motion.gyroYaw,

            ## 각도 정보
            "angleRoll": self.motion.angleRoll,
            "anglePitch": self.motion.anglePitch,
            "angleYaw": self.motion.angleYaw,
            
            # 고도 센서 정보들
            ## rpy 정보
            # "roll": self.altitude.roll,
            # "pitch": self.altitude.Pitch,
            # "yaw": self.altitude.Yaw,
        }
        
        return json.dumps(data)

    def send_command(self, command_set):
        '''
        드론에 내리는 명령을 받고 드론 컨트롤러에게 명령을 내려보냄

        Parameters:
        :command_set    json 형식으로 되어 있는 명령 집합
        '''

        command_set = json.loads(command_set)

        # take off 정보 추출
        if command_set["takeOff"] == "true" and self.on_flying is False:
            print("Take off")
            self.on_flying = True
            self.drone.sendTakeOff()
        elif command_set["takeOff"] == "false" and self.on_flying is True:
            print("Landing...")
            self.on_flying = False
            self.drone.sendLanding()

        # 조종 정보
        roll = 0
        pitch = 0
        yaw = 0
        throttle = 0
        # Color = LightFlagsDrone.BodyGreen.value
        # lightIntensity = 100

        # 조종 정보 추출
        try:
            if command_set["roll"] != "":
                roll = int(command_set["roll"])

            if command_set["pitch"] != "":
                pitch = int(command_set["pitch"])

            if command_set["yaw"] != "":
                yaw = int(command_set["yaw"])

            if command_set["throttle"] != "":
                throttle = int(command_set["throttle"])

            # if command_set["lightColor"] != "":
            #     lightColor = command_set["lightColor"]
            # if command_set["lightIntensity"] != "":
            #     lightIntensity = int(command_set["lightIntensity"])

            # if lightColor == "red":
            #     color = LightFlagsDrone.BodyRed.value
            # elif lightColor == "green":
            #     color = LightFlagsDrone.BodyGreen.value
            # elif lightColor == "blue":
            #     color = LightFlagsDrone.BodyBlue.value

            if command_set["lightColorR"] != "":
                self.lightColorR = int(command_set["lightColorR"])
            if command_set["lightColorG"] != "":
                self.lightColorG = int(command_set["lightColorG"])
            if command_set["lightColorB"] != "":
                self.lightColorB = int(command_set["lightColorB"])

        except ValueError as e:
            print("INVALID input:", e)
            return False

        # 디버깅
        print("Comamnd Set:")
        print("TakeOff: {}".format(command_set["takeOff"]))
        print("Roll: {}".format(roll))
        print("Pitch: {}".format(pitch))
        print("Yaw: {}".format(yaw))
        print("Throttle: {}".format(throttle))
        print("Light Color RGB: {} {} {}".format(command_set["lightColorR"], command_set["lightColorG"], command_set["lightColorB"]))
        # print("Light Intensity: {}".format(command_set["lightIntensity"]))

        # 드론에 명령
        self.drone.sendControlWhile(roll, pitch, yaw, throttle, 100)
        self.drone.sendLightManual(DeviceType.Drone, LightFlagsDrone.BodyRed.value, self.lightColorR)
        self.drone.sendLightManual(DeviceType.Drone, LightFlagsDrone.BodyGreen.value, self.lightColorG)
        self.drone.sendLightManual(DeviceType.Drone, LightFlagsDrone.BodyBlue.value, self.lightColorB)

        return True

    def motion_handler(self, motion):
        '''
        드론으로부터 모션 정보를 받으면 자동으로 실행되는 콜백 함수

        Parameters:
        :motion         모션 데이터가 담긴 객체
        '''
        
        # 디버깅을 위한 로그
        # print("--- Motion Data ---")
        # print("  Accel (x, y, z): {0}, {1}, {2}".format(motion.accelX, motion.accelY, motion.accelZ))
        # print("  Gyro  (r, p, y): {0}, {1}, {2}".format(motion.gyroRoll, motion.gyroPitch, motion.gyroYaw))
        # print("  Angle (r, p, y): {0}, {1}, {2}".format(motion.angleRoll, motion.anglePitch, motion.angleYaw))

        # 저장한 정보 업데이트
        self.motion = motion

    def altitude_handler(self, altitude):
        '''
        드론으로부터 고도 정보를 받으면 자동으로 실행되는 콜백 함수

        Parameters:
        :altitude       고도 데이터가 담긴 객체
        '''

        # 디버깅을 위한 로그
        print("--- RPY Data ---")
        print("  Roll / Pitch / Yaw: {0}, {1}, {2}".format(altitude.roll, altitude.pitch, altitude.yaw))

        # 저장한 정보 업데이트
        self.altitude = altitude
