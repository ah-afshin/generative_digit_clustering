from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_data_loader(B: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=transform
        )
    test_data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=transform
        )

    train_loader = DataLoader(train_data, batch_size=B, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=B, shuffle=False)

    return train_loader, test_loader
