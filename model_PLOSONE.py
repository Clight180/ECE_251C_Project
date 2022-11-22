import torch.nn as nn
import numpy as np
import config

Padding = 1
class DCNN(nn.Module):
    def __init__(self,channelsIn):
        super(DCNN,self).__init__()
        n_Flatten = 3136 if config.DWT_Input else 50176
        self.name = "PLOS_ONE"
        self.modelId = np.random.randint(100,999)

        self.A = nn.Sequential(
            nn.Conv2d(channelsIn, 16, (3,3), padding=Padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16)
        )

        self.B = nn.Sequential(
            nn.Conv2d(16, 32, (3,3), padding=Padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=Padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64)
        )

        self.D = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), padding=Padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding=Padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.f1 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(n_Flatten,256),
            nn.ReLU()
        )

        self.f2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.f3 = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x1 = self.A(x)
        x2 = self.B(x1)
        x3 = self.D(x2)
        x4 = self.f1(x3)
        # x5 = self.f2(x4)
        x6 = self.f3(x4)
        return x6