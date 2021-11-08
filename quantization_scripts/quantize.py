from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
import os
import csv

from quantize_neural_net import QuantizeNeuralNet
# from model_trainer import test_model
from data_loaders import data_loader


log_file_name = '../logs/Quantization_Log.csv'

fields = ['Model Name', 'Dataset', 'Original Test Accuracy',
          'Quantized Test Accuracy', 'Bits', 'Include 0', 'Seed', 'Author']


if __name__ == '__main__':
    
    # LeNet_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # default_transform is used for all pretrained models and Normalize is mandatory
    # see https://pytorch.org/vision/stable/models.html
    default_transform = transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),  
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                        ])
    
    # hyperparameter section
    author = 'Jinjie'
    seed = 0
    batch_size = 8  # batch_size used for quantization
    num_workers = 4
    bits = 1 
    data_set = 'ILSVRC2012'   # 'ILSVRC2012', 'CIFAR10', 'MNIST' 
    model_name = 'alexnet' # choose models 
    original_test_accuracy = None # set to None to run test on the original model.
    transform = default_transform
    include_0 = True
    ignore_layers = []
    alphabet_scalar = 1
    # end of hyperparameter section
    
    # load the model to be quantized
    model = getattr(torchvision.models, model_name)(pretrained=True) 
    model.eval()  # eval() is necessary 
    
    # load the data loader for training and testing
    train_loader, test_loader = data_loader(data_set, batch_size, transform, num_workers)
    
    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, batch_size, 
                                  train_loader, bits=bits,
                                  include_zero=include_0, 
                                  ignore_layers=ignore_layers,
                                  alphabet_scalar=alphabet_scalar)
    quantized_model = quantizer.quantize_network()

    if include_0:
        saved_model_name = f'batch{batch_size}_b{bits}_include0_scaler{alphabet_scalar}_ds{data_set}'
    else: 
        saved_model_name = f'batch{batch_size}_b{bits}_scaler{alphabet_scalar}_ds{data_set}'

    torch.save(quantized_model, os.path.join('../models/'+model_name, saved_model_name))
    
    # if not original_test_accuracy:
    #     print(f'\nEvaluting the original model to get its accuracy\n')
    #     predictions, labels = test_model(test_loader, model)
    #     original_test_accuracy = np.sum(predictions == labels) / len(labels)
    
    # print(f'\nEvaluting the quantized model to get its accuracy\n')
    # predictions, labels = test_model(test_loader, quantized_model)
    # test_accuracy = np.sum(predictions == labels) / len(labels)
    
    # print(f'The testing accuracy is: {test_accuracy}.')

    # with open(log_file_name, 'a') as f:
    #     csv_writer = csv.writer(f)
    #     row = [
    #         model_name, data_set, batch_size, 
    #         original_test_accuracy, test_accuracy, bits,
    #         include_0, seed, author
    #     ]
    #     csv_writer.writerow(row)

