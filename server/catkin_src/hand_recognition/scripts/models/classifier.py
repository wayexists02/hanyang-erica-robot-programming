import torch
import torch.nn as nn
from env import *


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(8, 12, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(12, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(16, 24, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8*8*24, 24),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(24, len(CAT)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
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

