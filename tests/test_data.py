import torch
from torch.utils.data import Dataset
import os.path
import pytest

from mnist_cc.data import corrupt_mnist
@pytest.mark.skipif(not os.path.exists("data/processed/train_target.pt") or not os.path.exists("data/processed/test_target.pt"), reason="Data files not found")
def test_data():
    # Derive expected lengths from processed dataset on disk
    train_target = torch.load("data/processed/train_target.pt")
    test_target = torch.load("data/processed/test_target.pt")
    expected_train = train_target.shape[0]
    expected_test = test_target.shape[0]

    trainset, testset = corrupt_mnist()
    assert len(trainset) == expected_train, f"Expected {expected_train} training samples, got {len(trainset)}"
    assert len(testset) == expected_test, f"Expected {expected_test} test samples, got {len(testset)}"
    assert len(trainset[0][0].shape) == 3, f"Expected training sample shape to have 3 dimensions, got {len(trainset[0][0].shape)}"
    assert len(testset[0][0].shape) == 3, f"Expected test sample shape to have 3 dimensions, got {len(testset[0][0].shape)}"
    # assert that all labels are represented
    train_labels = set()
    test_labels = set()
    for i in range(len(trainset)):
        train_labels.add(int(trainset[i][1]))
    for i in range(len(testset)):
        test_labels.add(int(testset[i][1]))
    assert train_labels == set(range(10))
    assert test_labels == set(range(10))