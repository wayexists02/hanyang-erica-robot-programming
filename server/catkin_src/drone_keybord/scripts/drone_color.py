#!/usr/bin/env python

import rospy
from drone_message.msg import DroneCommand

def publish_test():
    rospy.init_node('color_publisher')
    pub = rospy.Publisher('codrone_cmd', DroneCommand, queue_size=None)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        topic_msg = DroneCommand()

        topic_msg.takeOff = 'false'
        topic_msg.roll = '0'
        topic_msg.pitch = '0'
        topic_msg.yaw = '0'
        topic_msg.throttle = '0'
        topic_msg.lightColorR = str(input('ColorR : '))
        topic_msg.lightColorG = str(input('ColorG : '))
        topic_msg.lightColorB = str(input('ColorB : '))

        rospy.loginfo('published %s'%topic_msg)
        pub.publish(topic_msg)
        rate.sleep()

    rospy.loginfo('Terminated ')

if __name__ == '__main__':
    try:
        publish_test()
    except rospy.ROSInterruptException:
        pass
