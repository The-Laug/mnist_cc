import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

class MnistDataset(Dataset):
    """Custom Dataset for MNIST data."""

    def __init__(self, data_folder: str = "data", train: bool = True) -> None:
        if train:
            self.images = torch.load(f"{data_folder}/processed/train_images.pt")
            self.target = torch.load(f"{data_folder}/processed/train_target.pt")
            self.name = "Corrupted MNIST - Train"
        else:
            self.images = torch.load(f"{data_folder}/processed/test_images.pt")
            self.target = torch.load(f"{data_folder}/processed/test_target.pt")
            self.name = "Corrupted MNIST - Test"

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], int(self.target[idx])
    
def show_image_and_target(images: torch.Tensor, targets: torch.Tensor, show: bool = True) -> None:
    """Display images with their targets."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {targets[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    if show:
        plt.show()

def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics."""
    train_dataset = MnistDataset(data_folder=datadir, train=True)
    test_dataset = MnistDataset(data_folder=datadir, train=False)
    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    show_image_and_target(train_dataset.images[:25], train_dataset.target[:25], show=False)
    plt.savefig("mnist_images.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()



if __name__ == "__main__":
    app = typer.Typer()
    
    @app.command()
    def main(
        raw_dir: str = typer.Option("data/raw", help="Directory with raw data"),
        processed_dir: str = typer.Option("data/processed", help="Directory to save processed data"),
        pre_process: bool = typer.Option(False, "--pre-process", help="Run preprocessing on raw data"),
        stats: bool = typer.Option(False, "--stats", help="Generate dataset statistics"),
        datadir: str = typer.Option("data", help="Data directory for statistics"),
    ):
        """Process MNIST data and optionally generate statistics."""
        if pre_process:
            print("Preprocessing data...")
            preprocess_data(raw_dir, processed_dir)
            print("Preprocessing complete!")
        
        if stats:
            print("Generating statistics...")
            dataset_statistics(datadir)
            print("Statistics saved!")
        
        if not pre_process and not stats:
            print("No action specified. Use --pre-process or --stats flags.")
    
    app()
