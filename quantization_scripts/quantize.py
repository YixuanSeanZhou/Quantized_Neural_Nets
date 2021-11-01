import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from quantize_neural_net import QuantizeNeuralNet
from train_mlp import test_mlp, MLP
from data_loaders import load_data_mnist, load_data_fashion_mnist, load_data_kmnist

from train_conv2d import CNN

if __name__ == '__main__':
    batch_size = 32  # batch_size used for quantization
    num_workers = 4
    
    # load the model to be quantized
    # model = torch.load('../models/mnist_mlp.pt', map_location=torch.device('cpu'))
    # # load the data loader for training and testing
    # train_loader, _, test_loader = load_data_mnist(batch_size, train_ratio=1, 
    #                                             num_workers=num_workers)
    
    # model = torch.load('../models/fashion_mlp.pt', map_location=torch.device('cpu'))
    # train_loader, _, test_loader = load_data_fashion_mnist(batch_size, train_ratio=1, 
    #                                              num_workers=num_workers)

    model = torch.load('../models/conv2d_kmlp.pt', map_location=torch.device('cpu'))
    train_loader, _, test_loader = load_data_kmnist(batch_size, train_ratio=1, 
                                                    num_workers=num_workers)

    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, batch_size, train_loader, bits=3)
    quantized_mlp = quantizer.quantize_network()
    predictions, labels = test_mlp(test_loader, quantized_mlp)
    test_accuracy = np.sum(predictions == labels) / len(labels)
    print(f'The testing accuracy is: {test_accuracy}.')
