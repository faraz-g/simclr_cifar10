from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import Dataset


class SimCLRAugmentations:
    def __init__(self, size: int, jitter_strength: int = 1):
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * jitter_strength,
            contrast=0.8 * jitter_strength,
            saturation=0.8 * jitter_strength,
            hue=0.2 * jitter_strength,
        )

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transforms(x), self.train_transforms(x)


def get_cifar_dataset(root_folder: str, train: bool = True, aug: bool = True) -> Dataset:
    transform = SimCLRAugmentations(size=32) if aug else SimCLRAugmentations(size=32).test_transforms

    return datasets.CIFAR10(root=root_folder, train=train, transform=transform, download=True)
