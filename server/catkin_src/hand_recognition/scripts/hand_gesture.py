from models.classifier import Classifier
from env import *
import torch


class HandGestureRecognizer():

    def __init__(self):
        self.clf = Classifier().cuda()
        self.clf.load(CLF_CKPT_PATH)
        self.clf.eval()

    def __call__(self, img):
        with torch.no_grad():
            img = img.reshape(1, 3, HEIGHT, WIDTH)
            img = torch.FloatTensor(img).cuda()
            logps = self.clf(img)
            _, top_k = torch.exp(logps).topk(1, dim=1)
            
        return top_k.cpu().numpy()
