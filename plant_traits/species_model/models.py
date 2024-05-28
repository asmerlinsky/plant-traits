import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.efficientnet import (EfficientNet_V2_S_Weights,
                                             efficientnet_v2_s)
from torchvision.models.swin_transformer import Swin_V2_B_Weights, swin_v2_b


class SpeciesClassifier(nn.Module):
    """
    A classifier por different plant species, It has a lot of room for improvement given the huge amount of different labels.

    """

    def __init__(self, n_classes):
        super(SpeciesClassifier, self).__init__()

        # getting an efficient net and training a few final layers
        self.backbone = efficientnet_v2_s(weights=swin_v2_b)

        self.backbone.requires_grad_(False)

        self.backbone.features[-1][0] = nn.Conv2d(
            256, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.backbone.features[-1][1] = nn.BatchNorm2d(
            4096, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features=4096, out_features=8192),
            nn.GELU(),
            # nn.Dropout(.2),
            nn.Linear(in_features=8192, out_features=n_classes),
        )

    def forward(self, x_image):
        return self.backbone(x_image)

    def predict(self, x_image):
        output = self.forward(x_image)
        # output = nn.functional.softmax(output, dim=1)
        return torch.argmax(output, dim=1)

    def predict_prob(self, x_image):
        output = self.forward(x_image)
        return nn.functional.softmax(output, dim=1)
