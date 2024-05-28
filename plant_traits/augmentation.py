from torchvision import transforms

from plant_traits.constants import IMG_SIZE


def getTransforms():

    first_transform = [transforms.ToTensor()]

    aug_transforms = [
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.2, 0.8)),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(0.5, 0.4, 0.2, 0.05),
    ]

    preprocessing_transforms = [  # T.ToTensor(),
        transforms.Resize(size=IMG_SIZE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    aug_transformer = transforms.Compose(first_transform + aug_transforms)
    val_transformer = transforms.Compose(first_transform + preprocessing_transforms)
    return aug_transformer, val_transformer
