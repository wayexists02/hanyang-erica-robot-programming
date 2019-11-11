#!/usr/bin/env python

import picamera
import io
import rospy
import time


class PiCamera():

    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.start_preview()

        rospy.loginfo("PiCamera was opened.")
        time.sleep(2)

    def __del__(self):
        self.camera.stop_preview()
        self.camera.close()

    def capture(self):
        stream = io.BytesIO()

        self.camera.capture(stream, format="jpeg", resize=(224, 224))
        stream.seek(0)
        return stream

