import torch
import torch.nn as nn

from torchvision.models import vgg11_bn
from env import *

# CAT = NOTHING_CAT
# CAT = SIGN_CAT


class Classifier(nn.Module):

    def __init__(self, cat):
        super(Classifier, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 64

            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 32

            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 16

            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 8
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(3, 16, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 64

            nn.Conv2d(16, 16, (7, 7), stride=1, padding=3), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 32

            nn.Conv2d(16, 16, (7, 7), stride=1, padding=3), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 16

            nn.Conv2d(16, 16, (7, 7), stride=1, padding=3), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0), # 8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 8, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(8, 4, (5, 5), stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(4, 3, (5, 5), stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(8*8*32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(16, len(cat)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):

        x1 = self.features1(x)
        x2 = self.features2(x)

        x = torch.cat([x1, x2], dim=1)
        x_rec = self.decoder(x)

        x = x.view(x.size(0), -1)
        logps = self.classifier(x)

        return logps, x_rec

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

