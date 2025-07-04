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
