import torch
from torch import nn, optim
from hand_gesture import HandGestureRecognizer
from env import *
import cv2
import numpy as np


def main():
    model = HandGestureRecognizer()

    cap = cv2.VideoCapture(0)

    _input = None

    while _input != ord("e") and _input != ord("E"):
        ret, frame = cap.read()
        if ret is False:
            continue

        frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = (rgb_frame.astype(np.float32) - 128) / 256
        
        pred = model(rgb_frame)

        cv2.imshow("test", frame)
        _input = cv2.waitKey(100)


if __name__ == "__main__":
    with torch.no_grad():
        main()
