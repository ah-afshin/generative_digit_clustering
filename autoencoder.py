import torch as t
from torch import nn


class AutoEncoderMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), # 28*28 -> 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid() # each pixel ranges from 0 to 1
        )

    def forward(self, x: t.Tensor, encode_decode_only: str|None = None) -> t.Tensor:
        if encode_decode_only:
            match encode_decode_only.lower():
                case 'encode':
                    return self.encoder(x)
                case 'decode':
                    return self.decoder(x)
                case _:
                    raise ValueError(f'unknown encoder-decoder state: {encode_decode_only}')
        z = self.encoder(x)
        return self.decoder(z)


class SemiSupervisedAutoEncoderMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), # 28*28 -> 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid() # each pixel ranges from 0 to 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 64), # 64 here is actually a hyperparam, could be anything
            nn.ReLU(),
            nn.Linear(64, 10) # because we have 10 classes in MNIST dataset
        )
    
    def forward(self, x: t.Tensor, encode_decode_only: str|None = None) -> tuple[t.Tensor, t.Tensor]:
        if encode_decode_only:
            match encode_decode_only.lower():
                case 'encode':
                    return self.encoder(x)
                case 'decode':
                    return self.decoder(x)
                case 'classify':
                    return self.classifier(self.encoder(x))
                case _:
                    raise ValueError(f'unknown encoder-decoder state: {encode_decode_only}')
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        pred_label = self.classifier(z)
        return reconstructed, pred_label
