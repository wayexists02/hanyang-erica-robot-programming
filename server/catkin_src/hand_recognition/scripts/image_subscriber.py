import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from env import *
from image_viewer import ImageViewer


class ImageSubscriber():

    def __init__(self):
        self.cvbridge = CvBridge()
        self.img_buf = None
        self.image_subs = rospy.Subscriber("image", Image, callback=self._callback, queue_size=1)
        self.image_viewer = ImageViewer()

        self.test = rospy.get_param("/stop_param")

    def wait_for_image(self):
        if self.test is True:
            self.img_buf = self.image_viewer.get_test_camera()

        if self.img_buf is None:
            return None, None

        key = self.image_viewer.show(self.img_buf)

        img = self.img_buf
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(HEIGHT, WIDTH))
        img = (img.astype(np.float32) - 128) / 256

        self.img_buf = None

        return img, key

    def _callback(self, msg):
        cv_img = self.cvbridge.imgmsg_to_cv2(msg)
        self.img_buf = cv_img

