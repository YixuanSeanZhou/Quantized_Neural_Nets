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

log_file_name = '../logs/Quantization_Log.csv'


if __name__ == '__main__':

    # hyperparameter section
    bits_list = [4]
    scalar_list = [0.975]
    batch_size_list = [32, 64, 128] # batch_size used for quantization
    percentile_list = [1.0]   # quantile of weight matrix W
    num_workers = 8
    data_set = 'ILSVRC2012'   # 'ILSVRC2012', 'CIFAR10', 'MNIST' 
    model_name = 'vgg16' # choose models 
    include_0 = True
    ignore_layers = []
    retain_rate = 0.25
    author = 'Jinjie'
    seed = 0 

    # default_transform is used for all pretrained models and Normalize is mandatory
    # see https://pytorch.org/vision/stable/models.html
    transform = transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),  
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                        ])
    np.random.seed(seed)
    
    # NOTE: When using other network, just copy from pytorch website to here.
    # https://pytorch.org/vision/stable/models.html
    original_accuracy_table = {
        'alexnet': (.56522, .79066),
        'vgg16': (.71592, .90382),
        'resnet18': (.69758, .89078),
    }

    params = [(b, s, bs, per) for b in bits_list for s in scalar_list for 
                        bs in batch_size_list for per in percentile_list]


    # testing section
    for b, s, bs, per in params:
        batch_size = bs  
        bits = b  
        percentile = per
        alphabet_scalar = s  

        np.random.seed(seed)
        
        # load the model to be quantized from PyTorch resource
        model = getattr(torchvision.models, model_name)(pretrained=True) 
        model.eval()  # eval() is necessary 

        print(f'\nQuantizing {model_name} with bits: {bits}, include_0: {include_0}, scaler: {alphabet_scalar}, percentile: {percentile}, retain_rate: {retain_rate}, batch_size {batch_size}\n')
        
        # load the data loader for training and testing
        train_loader, test_loader = data_loader(data_set, batch_size, transform, num_workers)
        
        # quantize the neural net
        quantizer = QuantizeNeuralNet(model, batch_size, 
                                    train_loader, bits=bits,
                                    include_zero=include_0, 
                                    ignore_layers=ignore_layers,
                                    alphabet_scalar=alphabet_scalar,
                                    percentile=percentile,
                                    retain_rate=retain_rate)
        quantized_model = quantizer.quantize_network()

        if include_0:
            saved_model_name = f'batch{batch_size}_b{bits}_include0_scaler{alphabet_scalar}_percentile{percentile}_retain_rate{retain_rate}_ds{data_set}'
        else: 
            saved_model_name = f'batch{batch_size}_b{bits}_scaler{alphabet_scalar}_percentile{percentile}_retain_rate{retain_rate}_ds{data_set}'

        torch.save(quantized_model, os.path.join('../models/'+model_name, saved_model_name))

        topk = (1, 5)   # top-1 and top-5 accuracy
        
        if model_name in original_accuracy_table:
            print(f'\nUsing the original model accuracy from pytorch.\n')
            original_topk_accuracy = original_accuracy_table[model_name]
        else:
            print(f'\nEvaluting the original model to get its accuracy\n')
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
                bits, alphabet_scalar, percentile, include_0, retain_rate, seed, author
            ]
            csv_writer.writerow(row)
