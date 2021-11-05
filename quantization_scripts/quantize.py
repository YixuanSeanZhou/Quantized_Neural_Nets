from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
import os

import csv

import torchvision.models as models
from torchvision import transforms

from quantize_neural_net import QuantizeNeuralNet
from model_trainer import test_model

from data_loaders import data_loader, data_loader_miniimagenet

from models import LeNet5, CNN

log_file_name = '../logs/Quantization_Log.csv'

fields = ['Model Name', 'Dataset', 'Original Test Accuracy',
          'Quantized Test Accuracy', 'Bits', 'Include 0', 'Seed', 'Author']


def augment(x):
    return x.repeat(3, 1, 1)

if __name__ == '__main__':
    
    default_transform = transforms.Compose([transforms.ToTensor()])
    LeNet_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    AlexNet_transform = transforms.Compose([
                                            transforms.Resize((63, 63)),
                                            transforms.ToTensor(),
                                            augment
                                            ])

    # pretrained_transform is used for all pretrained models and Normalize is mandatory
    min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    # see https://pytorch.org/vision/stable/models.html
    pretrained_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Resize(min_img_size),   
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                         ])
    
    # hyperparameter section
    author = 'Yixuan Zhou'
    seed = 0
    batch_size = 32  # batch_size used for quantization
    num_workers = 4
    bits = 1
    data_set = 'MNIST'
    model_name = 'LeNet' # choose models trained by ourselves 
    original_test_accuracy = None # set to None to run test on the original model.
    transform = LeNet_transform
    include_0 = True
    # end of hyperparameter section

    transform = LeNet_transform
    
    batch_size = 32  # batch_size used for quantization
    num_workers = 4
    bits = 1
    
    # model_name = 'vgg16'
    # model_path = os.path.join('../models', model_name) # only needed for our trained models
    # ds_name =  'MiniImagenet'    # name of dataset, use names in following link 
    # https://pytorch.org/vision/stable/datasets.html#fashion-mnist

    # load the model to be quantized
    if model_name in ['vgg16', 'vgg16_bn']:  # add more models later
        model = getattr(torchvision.models, model_name)(pretrained=True) 
        model.eval()  # eval() is necessary 
    else:
        model_path = f'{model_name}_{data_set}.pt'
        model_path = os.path.join('../models', model_path)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        # choose dataset here
    
    if data_set == 'MiniImagenet':
        train_loader, _, test_loader = data_loader_miniimagenet(batch_size, transform, num_workers)
    else:
    # load the data loader for training and testing
        train_loader, _, test_loader = data_loader(data_set, batch_size, train_ratio=1, 
                                      num_workers=num_workers, 
                                      transform=transform
                                    )
    
    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, batch_size, 
                                  train_loader, bits=bits,
                                  include_zero=include_0)
    quantized_model = quantizer.quantize_network()
    
    if not original_test_accuracy:
        print(f'\nEvaluting the original model to get its accuracy\n')
        predictions, labels = test_model(test_loader, model)
        original_test_accuracy = np.sum(predictions == labels) / len(labels)
    
    print(f'\nEvaluting the quantized model to get its accuracy\n')
    predictions, labels = test_model(test_loader, quantized_model)
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

