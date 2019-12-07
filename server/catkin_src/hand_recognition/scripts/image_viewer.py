import cv2
import numpy as np


class ImageViewer():

    def __init__(self):
        self.win = cv2.namedWindow("Viewer")
        self.cap = None

    def get_test_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        ret, frame = self.cap.read()
        if ret < 0:
            print("ERROR in camera")
            return

        frame = cv2.resize(frame, dsize=(128, 128))

        return frame

    def show(self, img):
        cv2.imshow(self.win, img)
        key = cv2.waitKey(10)
        return key
