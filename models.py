import torch as t
from torch import nn
import torch.nn.functional as F


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


class VAEMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential( # h
            # h (after two layers) -> [μ, log(σ²)]
            nn.Flatten(),       # 28*28 -> 784
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 16)  # 16 is the size of compressed vec
        )
        self.mu = nn.Linear(16, 8)         # μ [8 is latent_dim, it's a hyperparam, could be anything]
        # learning a linear function for variance, could produce negative values for it,
        # which we don't want, so we learn log(var) and then use a exponential for reparameterization.
        self.log_var = nn.Linear(16, 8)    # log(σ²)
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            # nn.Sigmoid()                  # we kept this out of the model
        )                                   # results are raw logits
    
    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        std = t.exp(0.5 * log_var)  # this is a normal disterbution
        eps = t.randn_like(std)     # taking a sample of it
        return mu + eps * std       # z = μ + σ * ε (ε ~ N(0,1))
    
    def forward(self, x: t.Tensor, encode_decode_only: str|None = None) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        if encode_decode_only:
            match encode_decode_only.lower():
                case 'encode':
                    h = self.encoder(x)
                    mu, log_var = self.mu(h), self.log_var(h)
                    return self.reparameterize(mu, log_var)    # z = μ + σ * ε
                case 'decode':
                    return self.decoder(x)
                # case 'return-mu':
                #     h = self.encoder(x)
                #     return self.mu(h)
                case _:
                    raise ValueError(f'unknown encoder-decoder state: {encode_decode_only}')
        h = self.encoder(x)                         # h = f_encode(x)
        mu, log_var = self.mu(h), self.log_var(h)
        z = self.reparameterize(mu, log_var)        # z = μ + σ * ε
        # sampling is not learnable because it's not differentiable,
        # but we can differente this because it is a multiple of the selected sample.
        x_hat = self.decoder(z)                     # x^ = g_decode(z)
        return x_hat, mu, log_var                   # return mu & logvar 'cuz we need it for KL divergence loss



class CVAEMNIST(nn.Module):
    # this is really similar to `VAEMNIST` class
    # only new change are documented here
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential( # h <- f(x+y)
            nn.Linear(794, 400), # image_size + num_classes: 784 +10
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 16)  # 16 is the size of compressed vec
        )
        self.mu = nn.Linear(16, latent_dim)         # μ (compressed_dim -> latent_dim)
        self.log_var = nn.Linear(16, latent_dim)    # log(σ²)
        self.decoder = nn.Sequential(               # x^ <- g(z+y)
            nn.Linear(latent_dim + 10, 64),         # 10 is num_classes for MNIST
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
        )                                           # results are raw logits
    
    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        std = t.exp(0.5 * log_var)  # this is a normal disterbution
        eps = t.randn_like(std)     # taking a sample of it
        return mu + eps * std       # z = μ + σ * ε (ε ~ N(0,1))
    
    def forward(self, x: t.Tensor, label: t.Tensor, encode_decode_only: str|None = None) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        label_onehot = F.one_hot(label, num_classes=10).float()
        x = x.view(x.size(0), -1)                   # flatten
        x_cat = t.cat([x, label_onehot], dim=1)     # input
        
        if encode_decode_only:
            match encode_decode_only.lower():
                case 'encode':
                    h = self.encoder(x_cat)
                    mu, log_var = self.mu(h), self.log_var(h)
                    return self.reparameterize(mu, log_var)    # z = μ + σ * ε
                case 'decode':
                    # assume x is z here! (x_cat -> z_cat)
                    return self.decoder(x_cat)
                case 'return-mu':
                    h = self.encoder(x_cat)
                    return self.mu(h)
                case _:
                    raise ValueError(f'unknown encoder-decoder state: {encode_decode_only}')
        
        h = self.encoder(x_cat)                         # h = f_encode(x)
        mu, log_var = self.mu(h), self.log_var(h)
        z = self.reparameterize(mu, log_var)        # z = μ + σ * ε
        z_cat = t.cat([z, label_onehot], dim=1)     # again, labels are added
        x_hat = self.decoder(z_cat)                 # x^ = g_decode(z)
        return x_hat, mu, log_var                   # return mu & logvar 'cuz we need it for KL divergence loss
