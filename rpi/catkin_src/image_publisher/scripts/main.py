#!/usr/bin/env python

import rospy
from image_publisher import ImagePublisher 


def main():
    rospy.init_node("image_publisher_node", anonymous=False)

    img_pub = ImagePublisher()

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        cv_image = img_pub.get_cv_image()
        if img_pub.publish(cv_image) is False:
            rospy.signal_shutdown("Error in publishing...")
        rate.sleep()


if __name__ == "__main__":
    main()
