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
from helper_tools import test_accuracy
from data_loaders import data_loader

from my_models import LeNet5


log_file_name = '../logs/Quantization_Log.csv'

fields = ['Model Name', 'Dataset', 'Original Test Accuracy',
          'Quantized Test Accuracy', 'Bits', 'Include 0', 'Seed', 'Author']


if __name__ == '__main__':
    
    LeNet_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # default_transform is used for all pretrained models and Normalize is mandatory
    # see https://pytorch.org/vision/stable/models.html
    default_transform = transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),  
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                        ])
    
    hyper_bits = [4, 3]
    hyper_s = [8.5, 9, 8]

    hyperparams = [(b, s) for b in hyper_bits for s in hyper_s]

    for b, s in hyperparams:

        # hyperparameter section
        author = 'Yixuan'
        seed = 0
        batch_size = 128  # batch_size used for quantization
        num_workers = 8
        bits = b  # 1, 2, 3, 4
        data_set = 'ILSVRC2012'   # 'ILSVRC2012', 'CIFAR10', 'MNIST' 
        model_name = 'alexnet' # choose models 
        transform = default_transform
        include_0 = True
        ignore_layers = []
        retain_rate = 0.25
        alphabet_scalar = s   # 2, 3, 4, 5
        
        if model_name in {'LeNet', 'CNN'}:
            model_path = f'{model_name}_{data_set}.pt'
            model_path = os.path.join('../models', model_path)
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
        else:
            # load the model to be quantized
            model = getattr(torchvision.models, model_name)(pretrained=True) 
            model.eval()  # eval() is necessary 

        print(f'\nQuantizing {model_name} with bits: {bits}, include_0: {include_0}, scaler: {alphabet_scalar}\n')
        
        # load the data loader for training and testing
        train_loader, test_loader = data_loader(data_set, batch_size, transform, num_workers)
        
        # quantize the neural net
        quantizer = QuantizeNeuralNet(model, batch_size, 
                                    train_loader, bits=bits,
                                    include_zero=include_0, 
                                    ignore_layers=ignore_layers,
                                    alphabet_scalar=alphabet_scalar,
                                    retain_rate=retain_rate)
        quantized_model = quantizer.quantize_network()

        if include_0:
            saved_model_name = f'batch{batch_size}_b{bits}_include0_scaler{alphabet_scalar}_retain_rate{retain_rate}_ds{data_set}'
        else: 
            saved_model_name = f'batch{batch_size}_b{bits}_scaler{alphabet_scalar}_retain_rate{retain_rate}_ds{data_set}'

        torch.save(quantized_model, os.path.join('../models/'+model_name, saved_model_name))

        topk = (1, 5)   # top-1 and top-5 accuracy
        print(f'\n Evaluting the original model to get its accuracy\n')
        original_topk_accuracy = test_accuracy(model, test_loader, topk)
        print(f'Top-1 accuracy of {model_name} is {original_topk_accuracy[0]}.')
        print(f'Top-5 accuracy of {model_name} is {original_topk_accuracy[1]}.')
        
        print(f'\n Evaluting the quantized model to get its accuracy\n')
        topk_accuracy = test_accuracy(quantized_model, test_loader, topk)
        print(f'Top-1 accuracy of quantized {model_name} is {topk_accuracy[0]}.')
        print(f'Top-5 accuracy of quantized {model_name} is {topk_accuracy[1]}.')

        with open(log_file_name, 'a') as f:
            csv_writer = csv.writer(f)
            row = [
                model_name, data_set, batch_size, 
                original_topk_accuracy[0], topk_accuracy[0], original_topk_accuracy[1], topk_accuracy[1], 
                bits, alphabet_scalar, include_0, retain_rate, seed, author
            ]
            csv_writer.writerow(row)
