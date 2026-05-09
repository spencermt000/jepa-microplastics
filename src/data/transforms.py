from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def pretrain_transform(img_size: int = 224) -> transforms.Compose:
    """Augmentations for self-supervised pretraining (aggressive)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform(img_size: int = 224) -> transforms.Compose:
    """Minimal augmentations for probe training and evaluation."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def train_probe_transform(img_size: int = 224) -> transforms.Compose:
    """Light augmentations for probe training split."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
