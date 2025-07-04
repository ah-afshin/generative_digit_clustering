import torch as t
from torch import nn
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


def extract_encodings(model: nn.Module, dataloader: DataLoader) -> tuple[list, list]:
    encoded_data = []
    true_labels = []
    
    model.eval()
    with t.no_grad():
        for batch in dataloader:
            
            x, y = batch
            z = model(x, encode_decode_only='encode')
            encoded_data.append(z)
            true_labels.append(y)
    encoded_data = t.cat(encoded_data, dim=0)   # [N, B, latent_dim]   ->  [N, latent_dim]
    true_labels = t.cat(true_labels, dim=0)     # [N, B]               ->  [N]
    return encoded_data, true_labels
