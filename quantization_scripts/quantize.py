import torch
import torch.nn as nn
import torch.nn.functional as functional

from quantize_neural_net import QuantizeNeuralNet
from train_mlp import train_mlp, test_mlp, MLP
from data_loaders import load_data_mnist

if __name__ == '__main__':
    batch_size = 100

    # load the model to be quantized
    mlp = torch.load('../models/mlp.pt')

    # load the data loader for training and testing
    train_loader, val_loader, test_loader = load_data_mnist(batch_size)

    # specify the loss function for testing
    loss_function = nn.NLLLoss

    # quantize the neural net
    quantizer = QuantizeNeuralNet(mlp, batch_size, train_loader, 1)
    quantized_mlp = quantizer.quantize_network()

    test_mlp(quantized_mlp, test_loader, loss_function)
