from src.mnist_cc.model import MyAwesomeModel
import torch
import pytest
@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_forward_pass(batch_size):
    model = MyAwesomeModel()
    sample_input = torch.randn(batch_size, 1, 28, 28)  # batch size of 1, 1 channel, 28x28 image
    output = model(sample_input)
    assert output.shape == (batch_size, 10)  # assuming 10 classes for MNIST
    

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to be a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape'):
        model(torch.randn(1,1,28,29))