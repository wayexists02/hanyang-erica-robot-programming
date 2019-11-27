#!/usr/bin/env python

import picamera
from picamera.array import PiRGBArray
import io
import rospy
import time


RESOLUTION = (128, 128)

class PiCamera():

    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = 30
        
        self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
        self.stream = self.camera.capture_continuous(self.raw_capture, format="bgr")

        rospy.loginfo("PiCamera was opened.")
        time.sleep(2)

    def __del__(self):
        # self.camera.stop_preview()
        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

    def capture(self):
        # stream = io.BytesIO()

        # self.camera.capture(stream, format="jpeg")
        # stream.seek(0)
        # return stream

        for f in self.stream:
            if f is None:
                continue

            frame = f.array
            self.raw_capture.truncate(0)

            return frame

        return None

