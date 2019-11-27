#!/usr/bin/env python

from models.classifierVGG import ClassifierVGG
from image_subscriber import ImageSubscriber
from sign_publisher import SignPublisher
from hand_gesture import HandGestureRecognizer
from env import *
import rospy


def main():
    rospy.init_node("hand_gesture_node", anonymous=False)

    img_sub = ImageSubscriber()
    sign_pub = SignPublisher()
    model = HandGestureRecognizer()

    rate = rospy.Rate(5)

    rospy.loginfo("Hand recognizer started!")

    while not rospy.is_shutdown():
        if sign_pub.ready is False:
            continue

        img = img_sub.wait_for_image()
        if img is None:
            continue

        pred = model(img)
        rospy.loginfo("Prediction: {}".format(pred))
        
        sign_pub.send_action_command(int(pred))

        rate.sleep()


if __name__ == "__main__":
    main()