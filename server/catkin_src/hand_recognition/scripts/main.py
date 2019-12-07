#!/usr/bin/env python

import rospy

from std_msgs.msg import Bool

from image_subscriber import ImageSubscriber
from sign_publisher import SignPublisher
from hand_gesture import HandGestureRecognizer
from env import *

import cv2


def main():
    rospy.init_node("hand_gesture_node", anonymous=True)
    stop_pub = rospy.Publisher("stop", Bool, queue_size=1)

    img_sub = ImageSubscriber()
    sign_pub = SignPublisher()
    model = HandGestureRecognizer()

    rate = rospy.Rate(2)

    rospy.loginfo("Hand recognizer started!")

    while not rospy.is_shutdown():
        img, key = img_sub.wait_for_image()
        if key is None:
            pass
        elif key == ord("e") or key == ord("E"):
            cv2.destroyAllWindows()
            rospy.signal_shutdown("shutdown.")
        elif key == 27: # "ESC"
            stop_msg = Bool()
            stop_msg.data = True
            stop_pub.publish(stop_msg)
        elif key == 10: # "Enter"
            resume_msg = Bool()
            resume_msg.data = False
            stop_pub.publish(resume_msg)

        if img is None:
            rate.sleep()
            continue

        pred = model(img)
        sign_pub.send_command(int(pred))

        rate.sleep()

    exit(0)

if __name__ == "__main__":
    main()