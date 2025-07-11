import torch as t
from torch import nn



B = 32
LR = 2e-4
EPOCHS = 50
# ALPHA = 1.0
BETA = 3.5


def train(model:nn.Module, lr: float, epochs: int, data_loader: t.utils.data.DataLoader) -> None:
    optim = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0

        for batch in data_loader:
            image, _ = batch
            output = model(image)
            
            loss = criterion(output, image.view(image.size(0), -1)) # flatten the image
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_epoch_loss}")


def train_semisupervised(model:nn.Module, lr: float, epochs: int, alpha: float, beta: float, data_loader: t.utils.data.DataLoader) -> None:
    optim = t.optim.Adam(model.parameters(), lr)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0

        for batch in data_loader:
            x, y = batch
            x_recon, y_pred = model(x)

            loss_recon = criterion_recon(x_recon, x.view(x.size(0), -1))
            loss_class = criterion_class(y_pred, y)
            loss = alpha*loss_recon + beta*loss_class

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_epoch_loss}")


def train_vae(model:nn.Module, lr: float, epochs: int, beta: float, data_loader: t.utils.data.DataLoader) -> None:
    optim = t.optim.Adam(model.parameters(), lr)
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss(reduction='sum') # someone suggested this, idk why

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0

        for batch in data_loader:
            x, _ = batch
            x_recon, mu, logvar = model(x)

            recon_loss = criterion(x_recon, x.view(x.size(0), -1))
            kl_loss = -0.5 * t.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + beta * kl_loss) / x.size(0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_epoch_loss += loss.item()
        if epoch%5==4: print(f"Epoch {epoch+1} | loss: {total_epoch_loss}")


if __name__=="__main__":
    from autoencoder import VAEMNIST
    from utils import get_data_loader
    
    model = VAEMNIST()
    train_dl, _ = get_data_loader(B)
    train_vae(model, LR, EPOCHS, BETA, train_dl)
    t.save(model.state_dict(), "models/vae_autoencoder_3.pth")
