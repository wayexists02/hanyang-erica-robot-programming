#!/usr/bin/env python

import rospy
from pi_camera import PiCamera
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
import numpy as np
import cv2
import io


class ImagePublisher():

    def __init__(self):
        self.pi_camera = PiCamera()
        self.publisher = rospy.Publisher("/codrone_camera", Image, queue_size=10)
        self.bridge = CvBridge()
        self.pi_camera.open()

    def __del__(self):
        self.pi_camera.close()

    def get_cv_image(self):
        stream = self.pi_camera.capture()
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        cv_image = cv2.imdecode(data, 1)

        return cv_image

    def publish(self, cv_image):
        imgmsg = None
        
        try:
            imgmsg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            print(e)

        if imgmsg is None:
            rospy.logerror("CANNOT convert cv image into image message!")
            return False

        self.publisher.publish(imgmsg)
        rospy.loginfo("Image published!")

        return True

