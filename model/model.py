"""Models."""
import logging
import math
from typing import Dict

import torch
from torch import nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

class TraitDetector(nn.Module):

    def __init__(self, n_classes, train_features):
        super(TraitDetector, self).__init__()

        # The network is defined as a sequence of operations
        self.retinanet = retinanet_resnet50_fpn_v2(pretrained=True, weights_backbone=RetinaNet_ResNet50_FPN_V2_Weights)
        self.retinanet.requires_grad_(False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, train_features)

        self.tabular_nn = nn.Sequential(nn.Linear(in_features=train_features, out_features=train_features),
        nn.Dropout(p=.3),
        nn.Linear(in_features=train_features, out_features=train_features),
        nn.Dropout(p=.3),
        nn.Linear(in_features=train_features, out_features=train_features),
                      )

        self.merge_nn = nn.Sequential(nn.Linear(in_features=train_features, out_features=train_features),
        nn.Dropout(p=.3),
        nn.Linear(in_features=train_features, out_features=n_classes))




    # Specify the computations performed on the data
    def forward(self, x_image, x_row):
        x_image = self.retinanet(x_image)
        x_row = self.tabular_nn(x_row)

        return self.merge_nn(torch.cat(x_image, x_row))


    def predict(self, x_image, x_row):

        output = self.forward(x_image, x_row)

        return output
