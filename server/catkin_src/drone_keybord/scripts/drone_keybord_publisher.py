#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from drone_message.msg import DroneCommand  

def keybord():
	rospy.init_node("pup", anonymous=False)
	pup=rospy.Publisher("codrone_cmd", DroneCommand, queue_size=1)

	rate=rospy.Rate(10)
#
	# cap = cv2.VideoCapture(0)
	cv2.namedWindow('key_input', cv2.WINDOW_NORMAL)

	while not rospy.is_shutdown():
		msg=DroneCommand()
		# ret, key_input = cap.read()
		key_input = np.zeros((100, 100))
		cv2.imshow('key_input', key_input)
		key = cv2.waitKey(0)
		#print(key)
		if key == ord('t'):
			print ("input is t")
			msg.takeOff="true"

		if key == ord('l'):
			print ("input is l")
			msg.takeOff="false"

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

		pup.publish(msg)
		rospy.loginfo(msg)
		rate.sleep()


if __name__=='__main__':
	keybord()
