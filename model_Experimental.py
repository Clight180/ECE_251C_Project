import torch.nn as nn
import numpy as np
import config

kernelSize = (3,3)
Padding = 1

class DCNN(nn.Module):
    def __init__(self,channelsIn):
        super(DCNN,self).__init__()
        n_Flatten = 28800 if config.DWT_Input else 46208
        self.name = "Experimental"
        filt1 = 8
        filt2 = 16
        filt3 = 32
        self.modelId = np.random.randint(100,999)

        self.c1 = nn.Sequential(
            nn.Conv2d(channelsIn, filt1, kernelSize, padding=Padding),
            nn.MaxPool2d((3,3)),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(filt1, filt2, kernelSize, padding=Padding),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(filt2, filt3, kernelSize, padding=Padding),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        # self.c4 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c5 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c6 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c7 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c8 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c9 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c10 = nn.Sequential(
        #     nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt1),
        #     nn.ReLU()
        # )
        # self.c11 = nn.Sequential(
        #     nn.Conv2d(filt1, filt2, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt2),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2))
        # )
        # self.c12 = nn.Sequential(
        #     nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt2),
        #     nn.ReLU(),
        #     nn.Dropout(.1)
        # )
        # self.c13 = nn.Sequential(
        #     nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt2),
        #     nn.ReLU()
        # )
        # self.c14 = nn.Sequential(
        #     nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt2),
        #     nn.ReLU()
        # )
        # self.c15 = nn.Sequential(
        #     nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt2),
        #     nn.ReLU()
        # )
        # self.c16 = nn.Sequential(
        #     nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
        #     nn.BatchNorm2d(filt2),
        #     nn.ReLU()
        # )

        self.f1 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Dropout(.4),
            nn.Linear(n_Flatten,128),
            nn.ReLU()
        )

        self.f2 = nn.Sequential(
            nn.Dropout(.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        # x4 = self.c4(x3)
        # x5 = self.c5(x4)
        # x6 = self.c6(x5)
        # x7 = self.c7(x6)
        # x8 = self.c8(x7)
        # x9 = self.c9(x8)
        # x10 = self.c10(x9)
        # x11 = self.c11(x3)
        # x12 = self.c12(x11)
        # x13 = self.c13(x12)
        # x14 = self.c14(x13)
        # x15 = self.c15(x14)
        # x16 = self.c16(x15)
        x17 = self.f1(x3)
        x18 = self.f2(x17)
        return x18