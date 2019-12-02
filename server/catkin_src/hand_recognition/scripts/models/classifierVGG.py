import torch
import torch.nn as nn

from torchvision.models import vgg11_bn
from env import *


class ClassifierVGG(nn.Module):

    def __init__(self, cat):
        super(ClassifierVGG, self).__init__()
        
        self.cat = cat

        self.features = vgg11_bn(pretrained=True).features
        self.features.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Linear(4*4*512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, len(cat)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        self.features.eval()

        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def save(self, path, top_valid_acc):
        state_dict = {
            "state_dict": self.state_dict(),
            "top_valid_acc": top_valid_acc
        }
        torch.save(state_dict, path)

        print("Classifier was saved.")

    def load(self, path):
        state_dict = torch.load(path)

        self.load_state_dict(state_dict["state_dict"])
        self.top_valid_acc = state_dict["top_valid_acc"]

        print("Classifier was loaded")

