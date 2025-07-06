from sklearn.cluster import KMeans, DBSCAN
import torch as t


def cluster_kmeans(embeddings: t.Tensor, n_clusters: int = 10, random_state: int|None = None) -> t.Tensor:
    """KMeans clustering on embeddings.
    
    Args:
        embeddings [N, D]: encoders output
        n_clusters: num clusters (for MNIST it's 10)
        random_state: for reproducablity
    
    Returns:
        list of cluster labels for each sample
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embeddings_np)
    return t.tensor(kmeans.labels_) # all the vectors should be torch.Tensor


def cluster_dbscan(embeddings: t.Tensor, epsilon: int = 1) -> t.Tensor:
    """DBSCAN clustering on embeddings.
    
    Args:
        embeddings [N, D]: encoders output
        epsilon: max distance for dots to be neighbor
        random_state: for reproducablity
    
    Returns:
        list of cluster labels for each sample
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    dbscan = DBSCAN(epsilon)
    dbscan.fit(embeddings_np)
    return t.tensor(dbscan.labels_) # all the vectors should be torch.Tensor


if __name__=="__main__":
    from utils import extract_encodings
    from utils import get_data_loader
    from autoencoder import AutoEncoderMNIST
    from visualize import plot_tsne

    model = AutoEncoderMNIST()
    model.load_state_dict(t.load(f="models/autoencoder.pth"))
    _, test_dl = get_data_loader(32)
    encodec_vec, true_labels = extract_encodings(model, test_dl)
    
    kmean_pred = cluster_kmeans(encodec_vec, random_state=418)
    dbscan_pred = cluster_dbscan(encodec_vec, 1.95)
    plot_tsne(encodec_vec, kmean_pred, "Encodings k-mean pred labels")
    plot_tsne(encodec_vec, dbscan_pred, "Encodings DBSCAN pred labels")
    plot_tsne(encodec_vec, true_labels, "Encodings true labels")
