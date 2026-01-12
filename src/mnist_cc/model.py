from torch import nn
import torch
import pytest

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected input to be a 4D tensor, got {x.ndim}D tensor instead.")
        if x.shape[2:] != (28, 28):
            raise ValueError(f"Expected each sample to have shape [1, 28, 28], got {list(x.shape[1:])} instead.")
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    print("=== Model Test ===")
    model = MyAwesomeModel()
    print(model)
    x = torch.rand(1, 1, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
