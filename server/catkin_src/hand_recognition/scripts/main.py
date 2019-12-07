#!/usr/bin/env python

# from models.classifierVGG import ClassifierVGG
from image_subscriber import ImageSubscriber
from sign_publisher import SignPublisher
from hand_gesture import HandGestureRecognizer
from env import *
import rospy


def main():
    rospy.init_node("hand_gesture_node", anonymous=True)

    img_sub = ImageSubscriber()
    sign_pub = SignPublisher()
    model = HandGestureRecognizer()

    rate = rospy.Rate(2)

    rospy.loginfo("Hand recognizer started!")

    while not rospy.is_shutdown():
        img = img_sub.wait_for_image()
        if img is None:
            rate.sleep()
            continue

        pred = model(img)
        sign_pub.send_command(int(pred))

        rate.sleep()


if __name__ == "__main__":
    main()