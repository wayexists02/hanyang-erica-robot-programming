import torch
from env import *


class ClassifierTimeChain():

    def __init__(self, vgg):
        self.vgg = vgg

        self.x1 = None
        self.x2 = None

        self.x2_given_x1 = torch.eye(len(vgg.cat)).float().cuda()
        self.x2_given_x1 += 2
        self.x2_given_x1 /= torch.sum(self.x2_given_x1, dim=1, keepdim=True)

        self.x3_given_x1_x2 = torch.stack(
            [torch.eye(len(vgg.cat)).float().cuda() for i in range(len(vgg.cat))],
            dim=0
        )
        for i in range(len(vgg.cat)):
            self.x3_given_x1_x2[i] += 2
            self.x3_given_x1_x2[i, i, i] += 1
                    
        self.x3_given_x1_x2 /= torch.sum(self.x3_given_x1_x2, dim=-1, keepdim=True)

        # self.x_given_z = torch.eye(len(vgg.cat)).float().cuda()
        # self.x_given_z += 0.3
        # self.x_given_z /= torch.sum(self.x_given_z, dim=1, keepdim=True)

        print(self.x2_given_x1)
        print(self.x3_given_x1_x2)

    def __call__(self, x):
        with torch.no_grad():
            res = None

            logps = self.vgg(x)
            ps = torch.exp(logps)

            if self.x1 is None:
                res = ps
                self.x1 = res
            elif self.x2 is None:
                x1_pred = torch.argmax(self.x1, dim=1)
                res = self.x2_given_x1[x1_pred].view(*ps.size()) * ps
                self.x2 = res
            else:
                x1_pred = torch.argmax(self.x1, dim=1)
                x2_pred = torch.argmax(self.x2, dim=1)

                res = self.x3_given_x1_x2[x1_pred, x2_pred].view(*ps.size()) * ps

                self.x1 = self.x2
                self.x2 = res

        return res
