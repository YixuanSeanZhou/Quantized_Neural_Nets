from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

import csv

import torchvision.models as models
from torchvision import transforms

from quantize_neural_net import QuantizeNeuralNet
from train_conv2d import test
from data_loaders import load_data_mnist, load_data_fashion_mnist, load_data_kmnist, data_loader_dict

from models import LeNet5, CNN

log_file_name = '../logs/Quantization_Log.csv'

fields = ['Model Name', 'Dataset', 'Original Test Accuracy',
          'Quantized Test Accuracy', 'Bits', 'Include 0', 'Seed', 'Author']


def augment(x):
    return x.repeat(3, 1, 1)

if __name__ == '__main__':
    
    default_transform = [transforms.ToTensor()]
    LeNet_transform = [transforms.Resize((32, 32)), transforms.ToTensor()]
    AlexTransform = [
                    transforms.Resize((63, 63)),
                    transforms.ToTensor(),
                    augment
                    ]
    
    # hyperparameter section
    author = 'Yixuan Zhou'
    seed = 0
    batch_size = 32  # batch_size used for quantization
    num_workers = 4
    bits = 1
    data_set = 'mnist'
    model_name = 'LeNet'
    original_test_accuracy = 0.982
    transform = LeNet_transform
    include_0 = True
    # end of hyperparameter section
    
    
    dl = data_loader_dict[data_set]

    transform = LeNet_transform
    
    model_path = f'{model_name}_{data_set}.pt'
    model_path = os.path.join('../models', model_path)

    # load the model to be quantized
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # load the data loader for training and testing
    train_loader, _, test_loader = dl(batch_size, train_ratio=1, 
                                      num_workers=num_workers, 
                                      transform=transform
                                    )
    
    # model = torch.load('../models/fashion_mlp.pt', map_location=torch.device('cpu'))
    # train_loader, _, test_loader = load_data_fashion_mnist(batch_size, train_ratio=1, 
    #                                              num_workers=num_workers)

    # model = torch.load('../models/conv2d_kmlp.pt', map_location=torch.device('cpu'))
    # alexnet = models.alexnet()
    # train_loader, _, test_loader = load_data_kmnist(batch_size, train_ratio=1, 
    #                                                 num_workers=num_workers)

    # AlexTransform = [
    #                 transforms.Resize((63, 63)),
    #                 transforms.ToTensor(),
    #                 augment
    #                 ]

    # model = torch.load('../models/alex_fashion_mnist.pt', map_location=torch.device('cpu'))
    # train_loader, _, test_loader = load_data_fashion_mnist(batch_size, transform=AlexTransform, train_ratio=1, 
    #                                                         num_workers=num_workers)

    
    # model = torch.load('../models/lnet_mnist.pt', map_location=torch.device('cpu'))
    # train_loader, _, test_loader = load_data_mnist(batch_size, transform=LNetTransform, train_ratio=1, 
    #                                                         num_workers=num_workers)

    
    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, batch_size, 
                                  train_loader, bits=bits,
                                  include_zero=include_0)
    quantized_model = quantizer.quantize_network()
    predictions, labels = test(test_loader, quantized_model)
    test_accuracy = np.sum(predictions == labels) / len(labels)
    torch.save(quantized_model, f'../models/quantized_b{bits}_'+model_name)
    print(f'The testing accuracy is: {test_accuracy}.')

    with open(log_file_name, 'a') as f:
        csv_writer = csv.writer(f)
        row = [
            model_name, data_set, batch_size, 
            original_test_accuracy, test_accuracy, bits,
            include_0, seed, author
        ]
        csv_writer.writerow(row)

