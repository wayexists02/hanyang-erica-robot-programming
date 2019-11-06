#!/usr/bin/python

from google.cloud import vision
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


def detect_text(msg):
    client = vision.ImageAnnotatorClient()
    bridge = CvBridge()
    
    image = bridge.imgmsg_to_cv2(msg)
    _, buf = cv2.imencode(".jpg", image)
    image_byte = buf.tobytes()

    image = vision.types.Image(content=image_byte)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    rospy.loginfo("Texts:")
    for text in texts:
        rospy.loginfo("{}".format(text.description.encode("utf-8")))
