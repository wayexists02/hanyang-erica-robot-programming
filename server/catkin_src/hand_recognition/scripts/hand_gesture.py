from models.classifierVGG import ClassifierVGG
from env import *
import torch
import numpy as np
import rospy


class HandGestureRecognizer():

    def __init__(self):
        self.det = ClassifierVGG(NOTHING_CAT).cuda()
        self.det.load(NOTHING_CLF_CKPT_PATH)
        self.det.eval()

        self.clf = ClassifierVGG(SIGN_CAT).cuda()
        self.clf.load(SIGN_CLF_CKPT_PATH)
        self.clf.eval()

    def __call__(self, img):
        img = np.transpose(img, (2, 0, 1))
        img = img.reshape(1, *img.shape)

        # rospy.loginfo(img.shape)

        with torch.no_grad():
            img = torch.FloatTensor(img).cuda()
            logps = self.det(img)
            ps = torch.exp(logps)
            rospy.loginfo(ps)

            if ps[0, 1] > 0.5:
                logps = self.clf(img)
                _, top_k = torch.exp(logps).topk(1, dim=1)
                rospy.loginfo(SIGN_CAT[top_k])
            
                return top_k.cpu().numpy().squeeze()

        return -1
