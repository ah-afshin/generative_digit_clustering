import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch as t



def plot_tsne(embeddings: t.Tensor, labels: t.Tensor, title:str, perplexity: int = 30, n_samples: int = 2000) -> None:
    """t-SNE plot for embeddings.
    
    Args:
        embeddings [N, D]: encoded images vector
        labels [N]: true labels for coloring
        perplexity: t-SNE settings (usually 5 to 50)
        n_samples: num samples for plot
    """    
    embeddings_np = embeddings.detach().numpy()
    labels_np = labels.detach().numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings_np)

    X_scaled = X_scaled[:n_samples]
    labels_np = labels_np[:n_samples]

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    reduced = tsne.fit_transform(X_scaled)  # [n_samples, 2]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap='tab10', s=15)
    plt.colorbar(scatter, ticks=range(10), label='Digit')
    plt.title(f't-SNE Visualization of {title} (perplexity={perplexity})')
    plt.tight_layout()
    plt.show()


def plot_constructions(model, dl):
    """plot samples of reconstruction by model.
    
    Args:
        model [Module]: a trained reconstruction model
        dl [DataLoader]: a data loader of dataset samples
    """  
    model.eval()
    with t.no_grad():
        x, _ = next(iter(dl))
        x = x[:8]                                       # select 8 images only
        x_hat, _, _ = model(x)
        x_hat = t.sigmoid(x_hat.view(-1, 1, 28, 28))    # reconstruct as an image

        fig, axs = plt.subplots(2, 8, figsize=(15, 4))
        for i in range(8):
            axs[0, i].imshow(x[i].squeeze().numpy(), cmap='gray')
            axs[0, i].set_title("Original")
            axs[1, i].imshow(x_hat[i].squeeze().numpy(), cmap='gray')
            axs[1, i].set_title("Reconstructed")
            axs[0, i].axis("off")
            axs[1, i].axis("off")
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    from utils import extract_encodings
    from utils import get_data_loader
    from autoencoder import VAEMNIST

    model = VAEMNIST()
    model.load_state_dict(t.load(f="models/vae_autoencoder_2.pth"))
    _, test_dl = get_data_loader(32)
    encodec_vec, true_labels = extract_encodings(model, test_dl)
    plot_tsne(encodec_vec, true_labels, "Encodings")
    plot_constructions(model, test_dl)
