import torch as t
from torch import nn

from autoencoder import AutoEncoderMNIST
from utils import get_data_loader


B = 32
LR = 0.005
EPOCHS = 7


def train(model:nn.Module, lr: float, epochs: int, data_loader: t.utils.data.DataLoader) -> None:
    optim = t.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0

        for batch in data_loader:
            image, _ = batch
            output = model(image)
            
            loss = loss_func(output, image.view(image.size(0), -1)) # flatten the image
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_epoch_loss}")


if __name__=="__main__":
    model = AutoEncoderMNIST()
    train_dl, _ = get_data_loader(B)
    train(model, LR, EPOCHS, train_dl)
    t.save(model.state_dict(), "models/autoencoder.pth")
