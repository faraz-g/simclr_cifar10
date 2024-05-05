from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


def simclr_transform_pipeline(size: tuple[int, int], jitter_strength: int = 1) -> transforms.Compose:
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * jitter_strength,
        contrast=0.8 * jitter_strength,
        saturation=0.8 * jitter_strength,
        hue=0.2 * jitter_strength,
    )

    transform_pipeline = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ]
    )

    return transform_pipeline


class CIFARSimCLRDataLoader(DataLoader):
    def __init__(self, root_folder: str, transforms: transforms.Compose | None = None):
        self.dataset = datasets.CIFAR10(root=root_folder, train=True, download=True)
        self.transforms = simclr_transform_pipeline() if transforms is None else transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = self.dataset.__getitem__(idx)

        transform_one = self.transforms(img)
        transform_two = self.transforms(img)

        return transform_one, transform_two
