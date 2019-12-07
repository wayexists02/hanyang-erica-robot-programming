from models.classifierVGG import ClassifierVGG
from models.classifier_time_chain import ClassifierTimeChain
from env import *
import torch
import numpy as np


class HandGestureRecognizer():

    def __init__(self):
        self.det = ClassifierVGG(NOTHING_CAT).cuda()
        self.det.load(NOTHING_CLF_CKPT_PATH)
        self.det.eval()
        self.det = ClassifierTimeChain(self.det)

        self.clf = ClassifierVGG(SIGN_CAT).cuda()
        self.clf.load(SIGN_CLF_CKPT_PATH)
        self.clf.eval()
        self.clf = ClassifierTimeChain(self.clf)

    def __call__(self, img):
        img = np.transpose(img, (2, 0, 1))
        img = img.reshape(1, *img.shape)

        img = torch.FloatTensor(img).cuda()

        ps = self.det(img)
        pred = -1

        if ps[0, 1] > 0.5:
            print("There is a sign!")
            ps = self.clf(img)
            pred = torch.argmax(ps, dim=1).cpu().detach().numpy().squeeze()
            print("Prediction: " + str(pred))

        else:
            print("There is no hand sign.")

        return pred
