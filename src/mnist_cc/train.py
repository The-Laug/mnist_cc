import logging
import torch
from torch import nn
import matplotlib.pyplot as plt
from hydra import initialize, compose
from omegaconf import DictConfig
import hydra
from mnist_cc.data import corrupt_mnist
from mnist_cc.model import MyAwesomeModel
import random
import wandb

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> None:

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="lauge-arn-danmarks-tekniske-universitet-dtu",
        # Set the wandb project where this run will be logged.
        project="my-awesome-project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": cfg.hyperparameters.learning_rate,
            "architecture": "CNN",
            "dataset": "Corrupted MNIST",
            "epochs": cfg.hyperparameters.epochs,
            "batch_size": cfg.hyperparameters.batch_size,
            "seed": cfg.hyperparameters.seed,
        },
    )


    """Train a model on MNIST."""
    log.info("Training day and night")
    hyperparams = cfg.hyperparameters
    log.info(f"Hyperparameters: lr={hyperparams.learning_rate}, batch_size={hyperparams.batch_size}, epochs={hyperparams.epochs}")
    log.debug(f"Device: {DEVICE}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hyperparams.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(hyperparams.epochs):
        model.train()
        epoch_loss = 0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            epoch_loss += loss.item()

            if i % 100 == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
                run.log({"train_loss": loss.item(), "train_accuracy": accuracy, "epoch": epoch})
        
        avg_epoch_loss = epoch_loss / (i + 1)
        avg_epoch_acc = sum(statistics["train_accuracy"][-len(train_dataloader):]) / len(train_dataloader)
        log.info(f"Epoch {epoch + 1}/{hyperparams.epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_acc:.4f}")

    log.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    log.info("Model saved to models/model.pth")
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    run.log({"training_statistics": wandb.Image(fig)})
    artifact = wandb.Artifact("mnist-model", type="model")
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)
    run.finish()
    log.info("Training statistics saved to reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
