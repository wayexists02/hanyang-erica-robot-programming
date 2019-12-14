#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from drone_message.msg import DroneCommand
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool


roll = 0
throttle = 0
pitch = 0



def callback(data):
	global roll, throttle, pitch
	#rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
	circle_x, circle_y, radius = list(map(int, data.data.split(" ")))
	#circle_x = int(data.data[0])
	#circle_y = int(data.data[1])
	#radius = int(data.data[2])
	print(circle_x, circle_y, radius)
	if circle_x == 0 or circle_y==0:
		roll = 0
		throttle = 0
		pitch = 0

	else:
		roll = (circle_x - 64) * 0.2
		pitch = 0
		if circle_y <= 64:
			throttle = -(circle_y - 64) * 0.3
		else:
			throttle = -(circle_y - 64) * 0.15
		#pitch = 



def keyboard():

	rospy.init_node("pub", anonymous=False)
	rospy.Subscriber("circle_info", String, callback)
	pub=rospy.Publisher("/codrone_cmd", DroneCommand, queue_size=1)
	stop_pub=rospy.Publisher("stop", Bool, queue_size=1)
	pub_mission=rospy.ServiceProxy("mission", SetBool)

	#rospy.spin()

	rate=rospy.Rate(10)

	# cap = cv2.VideoCapture(0)
	cv2.namedWindow('key_input', cv2.WINDOW_NORMAL)

	while not rospy.is_shutdown():
		msg=DroneCommand()
		# ret, key_input = cap.read()
		key_input = np.zeros((100, 100))
		cv2.imshow('key_input', key_input)
		key = cv2.waitKey(10)
		#print(key)
		if key == ord('t'):
			print ("input is t")
			msg.takeOff="true"

		if key == ord('l'):
			print ("input is l")
			msg.takeOff="false"

		msg.roll = "%d" %roll
		msg.throttle = "%d" %throttle
		msg.pitch = "%d" %pitch

		if key == ord('q') or key == ord('Q'):
			print ("input is q")
			msg.yaw="-15"

		if key == ord('w') or key == ord('W'):
			print ("input is w")
			msg.pitch="15"

		if key == ord('e') or key == ord('E'):
			print ("input is e")
			msg.yaw="15"

		if key == ord('a') or key == ord('A'):
			print ("input is a")
			msg.roll="-15"

		if key == ord('s') or key == ord('S'):
			print ("input is s")
			msg.roll="0"
			msg.pitch="0"
			msg.throttle="0"


		if key == ord('d') or key == ord('D'):
			print ("input is d")
			msg.roll="15"


		if key == ord('z') or key == ord('Z'):
			print ("input is z")

		if key == ord('x') or key == ord('X'):
			print ("input is x")
			msg.pitch="-15"

		if key == ord('c') or key == ord('C'):
			print ("input is c")

		if key == ord('k') or key == ord('K'):
			print ("input is k")
			msg.throttle="-15"

		if key == ord('i') or key == ord('I'):
			print ("input is i")
			msg.throttle="15"

		if key == ord('m') or key == ord('M'):
			pub_mission(True)

		if key == ord('n') or key == ord('N'):
			pub_mission(False)

		if key == ord('b') or key == ord('B'):
			stop_pub.publish(Bool(True))

		pub.publish(msg)
		rospy.loginfo(msg.roll)
		rospy.loginfo(msg.pitch)
		rospy.loginfo(msg.throttle)
		rospy.loginfo(msg.takeOff)
		
		rate.sleep()

if __name__=='__main__':
	keyboard()
