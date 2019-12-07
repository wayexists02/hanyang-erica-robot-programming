import rospy
import actionlib
from drone_message.msg import DroneCommand
from std_msgs.msg import Bool


class SignPublisher():

    def __init__(self):
        self.sign_pub = rospy.Publisher("sign_cmd", DroneCommand, queue_size=1)
        self.mission_sub = rospy.Subscriber("mission", Bool, callback=self.mission_handle, queue_size=1)
        self.pitch = 0
        self.roll = 0
        self.yaw = 0

        self.msg = DroneCommand()

        self.on_mission = False

        self.current_direction = None

    def mission_handle(self, msg):
        if msg.data == True:
            self.on_mission = True
        else:
            self.on_mission = False

        # rospy.loginfo("ON MISSION" + str(self.on_mission))

    def send_command(self, pred):

        if self.on_mission is True:
            return

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
            self.current_direction = "left"
        elif pred == 5:
            self.current_direction = "right"
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
            elif self.current_direction == "left":
                self.msg.pitch = "0"
                self.msg.roll = "5"
                self.msg.yaw = "0"
            elif self.current_direction == "right":
                self.msg.pitch = "0"
                self.msg.roll = "-5"
                self.msg.yaw = "0"
            else:
                self.msg.pitch = "0"
                self.msg.yaw = "0"
                self.msg.roll = "0"

        self.msg.lightColorR = "255"
        self.msg.lightColorG = "0"
        self.msg.lightColorB = "0"

        self.sign_pub.publish(self.msg)
