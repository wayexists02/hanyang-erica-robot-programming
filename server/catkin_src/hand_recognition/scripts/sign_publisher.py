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

        self.act = None

        self.current_direction = []

        self.sign_action_clnt.wait_for_server()

    def send_action_command(self, pred):
        if self.ready is False:
            return

        self.ready = False

        self.act = DroneCommandGoal()

        takeoff_cmd = False
        
        if pred == 0:
            self.act.takeOff = "false"
            self.act.lightColorR = "255"
            self.act.lightColorG = "0"
            self.act.lightColorB = "0"
            takeoff_cmd = True
        elif pred == 1:
            self.act.takeOff = "true"
            self.act.lightColorR = "255"
            self.act.lightColorG = "0"
            self.act.lightColorB = "0"
            takeoff_cmd = True
        elif pred == 2 and not "forward" in self.current_direction:
            self._clear_direction_queue()
            self.current_direction.append("forward")
        elif pred == 3 and not "backward" in self.current_direction:
            self._clear_direction_queue()
            self.current_direction.append("backward")
        elif pred == 4 and not "yaw" in self.current_direction:
            self._clear_direction_queue()
            self.current_direction.append("yaw")
        elif pred == 5:
            self._clear_direction_queue()

        if takeoff_cmd is False:
            if "forward" in self.current_direction:
                self.act.pitch = "30"
            elif "backward" in self.current_direction:
                self.act.pitch = "-30"
            else:
                self.act.pitch = "0"

            if "yaw" in self.current_direction:
                self.act.yaw = "30"
            else:
                self.act.yaw = "0"

        self.act.lightColorR = "255"
        self.act.lightColorG = "0"
        self.act.lightColorB = "0"

        self.sign_action_clnt.send_goal(self.act, self.action_done)

    def action_done(self, state, result):
        rospy.loginfo("Action state: {}".format(state))
        rospy.loginfo("Result: {}".format(result))

        self.act.lightColorR = "0"
        self.act.lightColorG = "255"
        self.act.lightColorB = "0"
        self.sign_action_clnt.send_goal(self.act)

        del self.act

        self.ready = True

    def _clear_direction_queue(self):
        while len(self.current_direction) > 0:
            self.current_direction.pop()
