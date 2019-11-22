import torch
from torch import nn, optim
from models.classifier import Classifier
from models.classifierVGG import ClassifierVGG
from models.dataloader import DataLoader
from env import *
import cv2
import numpy as np


def main():
    det = ClassifierVGG(NOTHING_CAT).cuda()
    det.load(NOTHING_CLF_CKPT_PATH)

    clf = ClassifierVGG(SIGN_CAT).cuda()
    clf.load(SIGN_CLF_CKPT_PATH)

    cap = cv2.VideoCapture(1)

    _input = None

    while _input != ord("e") and _input != ord("E"):
        ret, frame = cap.read()
        if ret is False:
            continue

        frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.transpose(rgb_frame, (2, 0, 1))
        rgb_frame = rgb_frame.reshape(1, *rgb_frame.shape)

        rgb_frame = (rgb_frame.astype(np.float32) - 128) / 256

        logps = det(torch.FloatTensor(rgb_frame).cuda())
        ps = torch.exp(logps)
        print(ps)
        if ps[0, 1] > 0.5:
            print("Hand detected.")
            logps = clf(torch.FloatTensor(rgb_frame).cuda())
            ps = torch.exp(logps)
            cat = torch.argmax(ps, dim=1)

            print(SIGN_CAT[cat])

        cv2.imshow("test", frame)
        _input = cv2.waitKey(10)


if __name__ == "__main__":
    main()
