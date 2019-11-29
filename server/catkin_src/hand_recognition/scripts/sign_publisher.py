import rospy
import actionlib
from drone_message.msg import DroneCommandAction, DroneCommandGoal, DroneCommand


class SignPublisher():

    def __init__(self):
        # self.sign_pub = rospy.Publisher("sign_cmd", DroneCommand, queue_size=None)
        self.sign_action_clnt = actionlib.SimpleActionClient("codrone_cmd_action", DroneCommandAction)
        # self.cmd_queue = []
        self.ready = True
        self.pitch = 0
        self.roll = 0
        self.yaw = 0

        self.act = DroneCommandGoal()

        self.current_direction = None
        self.sign_action_clnt.wait_for_server()

    def send_action_command(self, pred):

        if self.ready is False:
            return

        self.ready = False

        self.act.takeOff = ""
        
        if pred == 0:
            self.act.takeOff = "false"
            self.current_direction = None
        elif pred == 1:
            self.act.takeOff = "true"
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
            self.act.pitch = "0"
            self.act.yaw = "0"
            self.act.roll = "0"
        else:
            if self.current_direction == "forward":
                self.act.pitch = "10"
                self.act.roll = "0"
                self.act.yaw = "0"
            elif self.current_direction == "backward":
                self.act.pitch = "-10"
                self.act.roll = "0"
                self.act.yaw = "0"
            elif self.current_direction == "yaw":
                self.act.pitch = "0"
                self.act.roll = "0"
                self.act.yaw = "-5"
            else:
                self.act.pitch = "0"
                self.act.yaw = "0"
                self.act.roll = "0"

        self.act.lightColorR = "255"
        self.act.lightColorG = "0"
        self.act.lightColorB = "0"

        self.sign_action_clnt.send_goal(self.act, self.action_done)

    def action_done(self, state, result):
        rospy.loginfo("Action state: {}".format(state))
        rospy.loginfo("Result: {}".format(result))

        self.act.takeOff = ""
        self.act.lightColorR = "0"
        self.act.lightColorG = "255"
        self.act.lightColorB = "0"
        self.sign_action_clnt.send_goal(self.act)

        self.ready = True
