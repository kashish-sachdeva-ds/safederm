"""Image preprocessing and augmentation pipeline for SafeDerm.

Defined once here so training notebooks and the eventual FastAPI serving
code use identical preprocessing -- prevents train/serve skew, where a
model trained on one pipeline is quietly served with a slightly different
one in production.
"""

from torchvision import transforms

# Backbone (ResNet, per ADR-002) is pretrained on ImageNet -- its early
# layers expect input normalized to ImageNet's per-channel mean/std.
# Computing custom stats from HAM10000 instead would mismatch what the
# pretrained weights expect, working against transfer learning rather
# than helping it.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224  # standard ResNet input size


def get_train_transforms() -> transforms.Compose:
    """Resize, augment, normalize. Used for training data only.

    Dermoscopic lesions have no canonical orientation -- a mole has no
    "right way up" -- so horizontal flip, vertical flip, and rotation are
    safe, realistic augmentations here. Mild color jitter accounts for
    lighting/device variation across the clinics that contributed to
    this dataset.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Resize and normalize only. Used for validation and test data.

    No augmentation -- ever. Val/test measure how the model performs on
    realistic, unmodified input. Augmenting them would make metrics
    reflect the augmentation, not the model.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
