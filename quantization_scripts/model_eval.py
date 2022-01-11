from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
import os
import csv
from datetime import datetime, timedelta

from quantize_neural_net import QuantizeNeuralNet
from helper_tools import test_accuracy
from data_loaders import data_loader

log_file_name = '../logs/Quantization_Log.csv'


if __name__ == '__main__':

    bits = [3, 4, 5]
    scalar_list = [1.41]
    mlp_scalar_list = [1.6] 
    cnn_scalar_list = [1.6] 
    batch_size_list = [4096] # batch_size used for quantization
    mlp_percentile_list = [1.0]   # quantile of weight matrix W
    cnn_percentile_list = [1.0]   # quantile of weight matrix W
    num_workers = 8
    data_set = 'ILSVRC2012'   # 'ILSVRC2012', 'CIFAR10', 'MNIST' 
    model_name = 'googlenet' # choose models 
    include_0 = True
    ignore_layers = []
    retain_rate = 0.25
    author = 'Yixuan'
    seed = 0 

    mlp_scalar = scalar_list[0]
    cnn_scalar = scalar_list[0]
    cnn_percentile = 1
    mlp_percentile = 1

    batch_size = 4096
    
    model_name = 'googlenet'

    transform = transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),  
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                        ])
    seed = 0 
    num_workers = 8
    topk = (1, 5)   # top-1 and top-5 accuracy
    np.random.seed(seed)

    train_loader, test_loader = data_loader(data_set, batch_size, transform, num_workers)

    file_name = os.path.join('../models/' + model_name, 'batch4096_b3_include0_mlpscalar1.41_cnnscalar1.41_mlppercentile1.0_cnnpercentile1.0_retain_rate0.25_dsILSVRC2012')

    model = torch.load(file_name)

    original_accuracy_table = {
        'alexnet': (.56522, .79066),
        'vgg16': (.71592, .90382),
        'resnet18': (.69758, .89078),
        'googlenet': (.69778, .89530),
        'resnet50': (.7613, .92862),
        'efficientnet_b1': (.7761, .93596),
        'efficientnet_b7': (.84122, .96908)
    }

    original_topk_accuracy = original_accuracy_table[model_name]

    start_time = datetime.now()

    print(f'\n Evaluting the quantized model to get its accuracy\n')
    topk_accuracy = test_accuracy(model, test_loader, topk)
    print(f'Top-1 accuracy of quantized {model_name} is {topk_accuracy[0]}.')
    print(f'Top-5 accuracy of quantized {model_name} is {topk_accuracy[1]}.')

    end_time = datetime.now()

    print(f'\nTime used for evaluation: {end_time - start_time}\n')

    with open(log_file_name, 'a') as f:
        csv_writer = csv.writer(f)
        row = [
            model_name, data_set, batch_size, 
            original_topk_accuracy[0], topk_accuracy[0], 
            original_topk_accuracy[1], topk_accuracy[1], 
            bits, mlp_scalar, cnn_scalar, 
            mlp_percentile, cnn_percentile, include_0, 
            retain_rate, seed, author
        ]
        # csv_writer.writerow(row)
