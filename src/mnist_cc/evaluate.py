import logging
import torch
import hydra
from omegaconf import DictConfig
from mnist_cc.data import corrupt_mnist
from mnist_cc.model import MyAwesomeModel

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def evaluate(cfg: DictConfig, model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model."""
    log.info("Evaluating like my life depended on it")
    log.info(f"Loading model from: {model_checkpoint}")
    log.debug(f"Device: {DEVICE}")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    log.info("Model loaded successfully")

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.hyperparameters.batch_size)
    log.info(f"Loaded test dataset with {len(test_set)} samples")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(test_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                log.debug(f"Evaluated batch {batch_idx}/{len(test_dataloader)}")
    
    accuracy = correct / total
    log.info(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate()
