import torch
from env import *


class ClassifierTimeChain():

    def __init__(self, vgg):
        self.vgg = vgg

        self.z1_given_x1 = None
        self.z2_given_x2 = None

        self.z1 = torch.ones((1, len(vgg.cat))).cuda()
        self.z1[0, 0] += 1
        self.z1 /= torch.sum(self.z1, dim=1, keepdim=True)

        self.z2_given_z1 = torch.eye(len(vgg.cat)).float().cuda()
        self.z2_given_z1 += 2
        self.z2_given_z1 /= torch.sum(self.z2_given_z1, dim=1, keepdim=True)

        self.z3_given_z2 = torch.stack(
            [torch.eye(len(vgg.cat)).float().cuda() for i in range(len(vgg.cat))],
            dim=0
        )
        for i in range(len(vgg.cat)):
            self.z3_given_z2[i] += 2
            self.z3_given_z2[i, i, i] += 1
            self.z3_given_z2[i] /= torch.sum(self.z3_given_z2[i], dim=1, keepdim=True)

        self.x_given_z = torch.eye(len(vgg.cat)).float().cuda()
        self.x_given_z += 0.3
        self.x_given_z /= torch.sum(self.x_given_z, dim=1, keepdim=True)

        print(self.initial_dist)
        print(self.transition_mat_prev_2)
        print(self.transition_mat_prev_1)

    def __call__(self, x):
        with torch.no_grad():
            res = None

            logps = vgg(x)
            ps = torch.exp(logps)

            if self.z1_given_x1 == None:
                self.z1_given_x1 = ps
                res = ps
            elif self.z2_given_x2 == None:
                self.z2_given_x2 = ps
                ps = torch.mm(self.prev_2, self.transition_mat_prev_2)

        return res
