import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights, efficientnet_v2_l
from torchvision.models.swin_transformer import Swin_V2_T_Weights, swin_v2_t


class SpeciesClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SpeciesClassifier, self).__init__()
        # The network is defined as a sequence of operations
        self.backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)
        self.backbone.requires_grad_(False)
        self.backbone.classifier[1] = nn.Linear(
            in_features=1280, out_features=n_classes
        )
        #
        # self.backbone = swin_v2_t(Swin_V2_T_Weights)
        # self.head = nn.Linear(in_features=768, out_features=n_classes)

        # self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # self.backbone.fc = nn.Linear(in_features=2048, out_features=n_classes)

    # Specify the computations performed on the data
    def forward(self, x_image):
        x = self.backbone(x_image)
        return nn.functional.softmax(x, dim=1)

    def predict(self, x_image):
        output = self.forward(x_image)
        return torch.argmax(output, dim=1)
