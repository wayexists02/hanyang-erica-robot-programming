#!/usr/bin/python

import rospy
from sensor_msgs.msg import Image
from ocr import detect_text


def main_fn():
    rospy.init_node("korean_ocr_node", anonymous=False)
    
    rospy.Subscriber("/image_in", Image, callback=detect_text, queue_size=None)
    
    rate = rospy.Rate(0.1)

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    main_fn()
