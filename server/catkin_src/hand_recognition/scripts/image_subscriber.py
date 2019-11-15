import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from env import *


class ImageSubscriber():

    def __init__(self):
        self.image_subs = rospy.Subscriber("/image_in", Image, callback=self._callback, queue_size=None)
        self.cvbridge = CvBridge()
        self.img_buf = []

    def wait_for_image(self):
        if len(self.img_buf) == 0:
            return None

        img = self.img_buf.pop(0)
        img = img[64:160, 64:160]
        img = cv2.resize(img, dsize=(HEIGHT, WIDTH))
        img = (img - 128) / 128
        return img

    def _callback(self, msg):
        cv_img = self.cvbridge.imgmsg_to_cv2(msg)
        cv_img = cv2.resize(cv_img, dsize=(HEIGHT, WIDTH))
        img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
        
        while len(self.img_buf) > 10:
            self.img_buf.pop(0)

        self.img_buf.append(cv_img)

