import matplotlib.pyplot as plt
import torch
import typer
from mnist_cc.model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions.
    args:
        model_checkpoint: Path to model checkpoint.
        figure_name: Name of the output figure file.
    returns:
        None

    """
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)
    # Load checkpoint onto the same device as the model to avoid device mismatch errors
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    # Replace final layer with identity to get embeddings
    model.fc = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    # Set no_grad for inference mode to save memory and computations
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            images = images.to(DEVICE)
            predictions = model(images)
            # Move to CPU for concatenation and downstream numpy conversion
            embeddings.append(predictions.cpu())
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings using PCA
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    # Now, run TSNE for final 2D visualization
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
