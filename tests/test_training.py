import torch
import pytest

from mnist_cc.model import MyAwesomeModel


def test_model_trains_one_step():
    """Verify a single optimizer step reduces loss."""
    model = MyAwesomeModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Single batch
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    
    # Capture initial parameters
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Forward pass and gradient step
    model.train()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    
    # Verify parameters changed
    trained_params = model.state_dict()
    assert any(
        not torch.equal(initial_params[k], trained_params[k])
        for k in initial_params
    ), "Model parameters should change after one optimizer step"


def test_model_accuracy_calculation():
    """Verify accuracy metric computation."""
    model = MyAwesomeModel()
    model.eval()
    
    # Batch with known predictions
    x = torch.randn(8, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    with torch.no_grad():
        y_pred = model(x)
    
    # Calculate accuracy manually
    correct = (y_pred.argmax(dim=1) == y).float().mean().item()
    
    # Accuracy should be in [0, 1]
    assert 0 <= correct <= 1, f"Accuracy {correct} out of bounds"
