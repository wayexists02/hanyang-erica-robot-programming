#!/usr/bin/env python

import rospy
from pi_camera import PiCamera
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class ImagePublisher():

    def __init__(self):
        self.pi_camera = PiCamera()
        self.publisher = rospy.Publisher("/codrone_camera", Image, queue_size=10)
        self.bridge = CvBridge()

    def get_cv_image(self):
        from PIL import Image

        image_bytes = self.pi_camera.capture()
        image = Image.open(image_bytes).convert("RGB")
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

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

