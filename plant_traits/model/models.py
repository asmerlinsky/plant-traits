"""Models."""

import logging
import math
from typing import Dict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights, efficientnet_v2_s

from plant_traits.species_model.models import SpeciesClassifier


class TraitDetector(nn.Module):

    def __init__(self, n_classes, train_features, species_weights_path):
        super(TraitDetector, self).__init__()

        # The network is defined as a sequence of operations
        # self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # self.resnet.requires_grad_(False)
        # self.backbone.fc = nn.Linear(in_features=2048, out_features=train_features)

        self.backbone = SpeciesClassifier(n_classes)
        self.backbone.load_state_dict(torch.load(species_weights_path))

        self.tabular_nn = nn.Sequential(
            nn.Linear(in_features=train_features, out_features=train_features),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(in_features=train_features, out_features=train_features),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(in_features=train_features, out_features=train_features),
            nn.GELU(),
        )

        self.merge_nn = nn.Sequential(
            nn.Linear(in_features=train_features, out_features=train_features),
            # nn.Dropout(p=0.3),
            nn.GELU(),
            nn.Linear(in_features=train_features, out_features=n_classes),
        )

    # Specify the computations performed on the data
    def forward(self, x_image, x_row):
        speccies_pred = self.backbone(x_image)
        x_row = self.tabular_nn(x_row)

        return self.merge_nn(torch.cat((x_image, x_row), axis=1))

    def predict(self, x_image, x_row):

        output = self.forward(x_image, x_row)

        return output


class StratifiedTraitDetector(nn.Module):

    def __init__(
        self,
        n_classes,
        train_features,
        groups_dict: dict,
        species_weights_path,
        species_df,
        topk,
    ):
        super(StratifiedTraitDetector, self).__init__()

        # The network is defined as a sequence of operations
        self.backbone = SpeciesClassifier(n_classes)
        self.backbone.requires_grad_(False)

        self.species_tensor = torch.from_numpy(species_df.values)

        self.topk = topk
        self.topk_W = torch.nn.Parameter(torch.ones(self.topk, dtype=torch.long))

        # self.resnet.train = lambda x: True
        #
        # self.resnet.training = False
        self.var_groups = list(groups_dict.keys())

        net_elements = 0
        for group, elements in groups_dict.items():

            net_elements += len(elements)
            if len(elements) > 1:
                self.__setattr__(
                    group,
                    nn.Sequential(
                        nn.Linear(
                            in_features=len(elements), out_features=train_features
                        ),
                        nn.GELU(),
                        # nn.Dropout(p=0.1),
                        nn.Linear(
                            in_features=train_features, out_features=len(elements)
                        ),
                        nn.GELU(),
                    ),
                )
            else:
                self.__setattr__(group, nn.Identity())

        self.merge_features_nn = nn.Sequential(
            nn.Linear(in_features=net_elements, out_features=int(net_elements / 2)),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(in_features=int(net_elements / 2), out_features=n_classes),
            nn.LeakyReLU(),
        )

        self.merge_final = nn.Sequential(
            nn.Linear(in_features=2 * n_classes, out_features=n_classes)
        )

    # Specify the computations performed on the data
    def forward(self, x_image, x_row_dict):
        sp_probs = self.backbone.predict(x_image)

        topk_predictions = self.species_tensor[sp_probs.topk(self.topk)[1]]

        species_prediction = torch.mul(self.topk_W, topk_predictions)

        group_tensors = [x_image]
        for group in self.var_groups:
            group_tensors.append(self.__getattr__(group)(x_row_dict[group]))

        merged_features = self.merge_features_nn(torch.cat(group_tensors, axis=1))

        return self.merge_features_nn(
            torch.cat([merged_features, species_prediction], axis=1)
        )

    def predict(self, x_image, x_row):

        output = self.forward(x_image, x_row)

        return output
