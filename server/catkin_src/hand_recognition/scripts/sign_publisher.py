import rospy
import actionlib
from drone_message.msg import DroneCommand


class SignPublisher():

    def __init__(self):
        self.sign_pub = rospy.Publisher("sign_cmd", DroneCommand, queue_size=1)
        self.pitch = 0
        self.roll = 0
        self.yaw = 0

        self.msg = DroneCommand()

        self.current_direction = None

    def send_command(self, pred):

        self.msg.takeOff = ""
        
        if pred == 0:
            self.msg.takeOff = "false"
            self.current_direction = None
        elif pred == 1:
            self.msg.takeOff = "true"
            self.current_direction = None
        elif pred == 2:
            self.current_direction = "forward"
        elif pred == 3:
            self.current_direction = "backward"
        elif pred == 4:
            self.current_direction = "yaw"
        elif pred == 5:
            self.current_direction = None
        else:
            pass

        if self.current_direction is None:
            self.msg.pitch = "0"
            self.msg.yaw = "0"
            self.msg.roll = "0"
        else:
            if self.current_direction == "forward":
                self.msg.pitch = "10"
                self.msg.roll = "0"
                self.msg.yaw = "0"
            elif self.current_direction == "backward":
                self.msg.pitch = "-10"
                self.msg.roll = "0"
                self.msg.yaw = "0"
            elif self.current_direction == "yaw":
                self.msg.pitch = "0"
                self.msg.roll = "0"
                self.msg.yaw = "-5"
            else:
                self.msg.pitch = "0"
                self.msg.yaw = "0"
                self.msg.roll = "0"

        self.msg.lightColorR = "255"
        self.msg.lightColorG = "0"
        self.msg.lightColorB = "0"

        self.sign_pub.publish(self.msg)
