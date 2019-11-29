from models.classifierVGG import ClassifierVGG
from env import *
import torch
import numpy as np
import rospy


class HandGestureRecognizer():
    """
    손인식 모델을 관리하는 객체
    """

    def __init__(self):
        # 손이 있는지 없는지 판단하는 classifier 생성 및 로드
        # self.det = ClassifierVGG(NOTHING_CAT).cuda()
        # self.det.load(NOTHING_CLF_CKPT_PATH)
        self.det = torch.load(NOTHING_CLF_CKPT_PATH)["model"]
        self.det.eval()

        # 손이 있다면, 몇 개의 손가락이 있는지 분류하는 classifier 생성 및 로드
        # self.clf = ClassifierVGG(SIGN_CAT).cuda()
        # self.clf.load(SIGN_CLF_CKPT_PATH)
        self.clf = torch.load(SIGN_CLF_CKPT_PATH)["model"]
        self.clf.eval()

    def __call__(self, img):
        """
        추론 함수

        Arguments:
        ----------
        img : RGB 이미지 1개

        Returns:
        --------
        res : 추론 결과 (-1은 손이 인식되지 않은 것)
        """

        # 이미지를 규격(?)에 맞춰줌
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
