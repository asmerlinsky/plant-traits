"""Models."""

import logging
import math
from typing import Dict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class TraitDetector(nn.Module):

    def __init__(self, n_classes, train_features):
        super(TraitDetector, self).__init__()

        # The network is defined as a sequence of operations
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.requires_grad_(False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=train_features)
        # self.resnet.train = lambda x: True
        #
        # self.resnet.training = False

        self.tabular_nn = nn.Sequential(
            nn.Linear(in_features=train_features, out_features=train_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=train_features, out_features=train_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=train_features, out_features=train_features),
        )

        self.merge_nn = nn.Sequential(
            nn.Linear(in_features=2 * train_features, out_features=train_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=train_features, out_features=n_classes),
        )

    # Specify the computations performed on the data
    def forward(self, x_image, x_row):
        x_image = self.resnet(x_image)
        x_row = self.tabular_nn(x_row)

        return self.merge_nn(torch.cat((x_image, x_row), axis=1))

    def predict(self, x_image, x_row):

        output = self.forward(x_image, x_row)

        return output
