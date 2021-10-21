import torch
import torch.nn as nn
import torch.nn.functional as functional

from quantize_neural_net import QuantizeNeuralNet
from train_mlp import train_mlp, test_mlp, MLP
from data_loaders import load_data_mnist

if __name__ == '__main__':
    batch_size = 50

    mlp = torch.load('../models/mlp.pt')
    train_loader, test_loader = load_data_mnist(batch_size)

    loss_function = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam

    quantizer = QuantizeNeuralNet(mlp, batch_size, train_loader, 5)
    quantized_mlp = quantizer.quantize_network()

    test_mlp(quantized_mlp, test_loader, nn.CrossEntropyLoss)
