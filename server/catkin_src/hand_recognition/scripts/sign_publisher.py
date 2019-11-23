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

        self.sign_action_clnt.wait_for_server()

    def send_action_command(self, pred):
        if self.ready is False:
            return

        self.ready = False

        act = DroneCommandGoal()
        
        if pred == 0:
            act.takeOff = "false"
            act.lightColorR = "255"
            act.lightColorG = "0"
            act.lightColorB = "0"
        elif pred == 1:
            act.takeOff = "true"
            act.lightColorR = "255"
            act.lightColorG = "0"
            act.lightColorB = "0"
        elif pred == 2 and self.pitch < 30:
            act.pitch = "30"
            self.pitch += 30
        elif pred == 3 and self.pitch > -30:
            act.pitch = "-30"
            self.pitch -= 30

        self.sign_action_clnt.send_goal(act, self.action_done)

    def action_done(self, state, result):
        rospy.loginfo("Action state: {}".format(state))
        rospy.loginfo("Result: {}".format(result))

        act = DroneCommandGoal()
        act.lightColorR = "0"
        act.lightColorG = "255"
        act.lightColorB = "0"
        self.sign_action_clnt.send_goal(act)

        self.ready = True

    # def send_command(self, pred):
    #     if self.ready is False:
    #         return

    #     self.ready = False

    #     # print(type(pred), pred)

    #     cmd = DroneCommand()
        
    #     if pred == 0:
    #         cmd.takeOff = "false"
    #         cmd.lightColorR = "255"
    #         cmd.lightColorG = "0"
    #         cmd.lightColorB = "0"
    #     elif pred == 1:
    #         cmd.takeOff = "true"
    #         cmd.lightColorR = "255"
    #         cmd.lightColorG = "0"
    #         cmd.lightColorB = "0"
    #     elif pred == 2 and self.pitch != "30":
    #         cmd.pitch = "30"
    #         self.pitch = "30"
    #     elif pred == 3 and self.pitch != "-30":
    #         cmd.pitch = "-30"
    #         self.pitch = "-30"

    #     self.sign_pub.publish(cmd)
    #     rospy.Timer(rospy.Duration(2), self.timer, oneshot=True)

    # def timer(self, e):
    #     cmd = DroneCommand()
    #     cmd.lightColorR = "0"
    #     cmd.lightColorG = "255"
    #     cmd.lightColorB = "0"

    #     self.sign_pub.publish(cmd)
    #     self.ready = True
