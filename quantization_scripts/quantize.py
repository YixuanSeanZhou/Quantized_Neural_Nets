import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from quantize_neural_net import QuantizeNeuralNet
from train_conv2d import test, CNN
from data_loaders import load_data_mnist, load_data_fashion_mnist, load_data_kmnist


if __name__ == '__main__':
    batch_size = 32  # batch_size used for quantization
    num_workers = 4
    bits = 3
    dl = load_data_fashion_mnist
    model_name = 'conv2d_fashion.pt'
    model_path = os.path.join('../models', model_name)
    # load the model to be quantized
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # load the data loader for training and testing
    train_loader, _, test_loader = dl(batch_size, train_ratio=1, 
                                                num_workers=num_workers)
    
    # model = torch.load('../models/fashion_mlp.pt', map_location=torch.device('cpu'))
    # train_loader, _, test_loader = load_data_fashion_mnist(batch_size, train_ratio=1, 
    #                                              num_workers=num_workers)

    # model = torch.load('../models/conv2d_kmlp.pt', map_location=torch.device('cpu'))
    # train_loader, _, test_loader = load_data_kmnist(batch_size, train_ratio=1, 
    #                                                 num_workers=num_workers)

    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, batch_size, train_loader, bits=bits)
    quantized_model = quantizer.quantize_network()
    predictions, labels = test(test_loader, quantized_model)
    test_accuracy = np.sum(predictions == labels) / len(labels)
    torch.save(quantized_model, f'../models/quantized_b{bits}_'+model_name)
    print(f'The testing accuracy is: {test_accuracy}.')
