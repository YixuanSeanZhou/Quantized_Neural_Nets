from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
import os
import csv
import gc
import glob
from datetime import datetime, timedelta

from quantize_neural_net import QuantizeNeuralNet
from correct_bias_last_layer import CorrectBiasLastLayer
from helper_tools import test_accuracy, get_param_from_model_name
from data_loaders import data_loader

log_file_name = '../logs/Quantization_Log_Bias_Correct.csv'




if __name__ == '__main__':

    # hyperparameter section

    batch_size_list = [2048, 1024, 4096]
            # 1024, 512, 256, 128, 64, 32] # batch_size used for quantization
    num_workers = 8
    data_set = 'ILSVRC2012'   # 'ILSVRC2012', 'CIFAR10', 'MNIST' 
    include_0 = True
    ignore_layers = []
    retain_rate = 0.25
    author = 'Yixuan'
    seed = 1
    model_name = 'resnet18'
    quantized_dir_name = '../adhoc_models/ref_models/resnet18/'
    # quantized_file_name = '../models/resnet50/batch512_b4_include0_scaler1.81_percentile1.0_retain_rate0.25_dsILSVRC2012'

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
        'googlenet': (.69778, .89530),
        'resnet50': (.7613, .92862),
        'efficientnet_b1': (.7761, .93596),
        'efficientnet_b7': (.84122, .96908),
        'mobilenet_v2': (.71878, .90286)
    }

    
    files = glob.glob(f'{quantized_dir_name}/*')
    params = [(bs, f) 
                for bs in batch_size_list
                for f in files
                ]

    # testing section
    for bs, f in params:
        bc_batch_size = bs  
        
        batch_size, bits, mlp_scalar, cnn_scalar = get_param_from_model_name(f)

        mlp_percentile = 1.0
        cnn_percentile = 1.0

        # TODO: update the quantize_class

        np.random.seed(seed)

        # load the model to be quantized from PyTorch resource
        model = getattr(torchvision.models, model_name)(pretrained=True) 
        model.eval()  # eval() is necessary 

        quantized_model = torch.load(f)
        quantized_model.eval()

        print(f'\nCorrecting Bias {model_name} with bc_batch_size {bc_batch_size}, with bits: {bits}, include_0: {include_0}, mlp_scalar: {mlp_scalar}, cnn_scalar: {cnn_scalar}, mlp_percentile: {mlp_percentile}, cnn_percentile: {cnn_percentile} retain_rate, {retain_rate}, batch_size {batch_size}\n')
        
        # load the data loader for training and testing
        train_loader, test_loader = data_loader(data_set, bc_batch_size, transform, num_workers)
        
        # correct bias for the neural net
        corrector = CorrectBiasLastLayer(
                                     model, 
                                     quantized_model,
                                     bc_batch_size, 
                                     train_loader, 
                                     mlp_bits=bits,
                                     cnn_bits=bits,
                                     include_zero=include_0, 
                                     ignore_layers=ignore_layers,
                                     mlp_alphabet_scalar=mlp_scalar,
                                     cnn_alphabet_scalar=cnn_scalar,
                                     mlp_percentile=mlp_percentile,
                                     cnn_percentile=cnn_percentile,
                                     retain_rate=retain_rate,
                                     )
        start_time = datetime.now()

        quantized_model = corrector.correct_bias_for_last_layer_network()

        end_time = datetime.now()

        print(f'\nTime used for correction: {end_time - start_time}\n')

        if include_0:
            saved_model_name = f'batch{batch_size}_b{bits}_include0_mlpscalar{mlp_scalar}_cnnscalar{cnn_scalar}_mlppercentile{mlp_percentile}_cnnpercentile{cnn_percentile}_retain_rate{retain_rate}_ds{data_set}'
        else: 
            saved_model_name = f'batch{batch_size}_b{bits}_mlpscalar{mlp_scalar}_cnnscalar{cnn_scalar}_mlppercentile{mlp_percentile}_cnnpercentile{cnn_percentile}_retain_rate{retain_rate}_ds{data_set}'
        
        
        saved_model_name += '_skip_last_layer'
        saved_model_name += '_bias_corrected'
        torch.save(quantized_model, os.path.join('../adhoc_models/'+model_name, saved_model_name))

        topk = (1, 5)   # top-1 and top-5 accuracy
        
        if model_name in original_accuracy_table:
            print(f'\nUsing the original model accuracy from pytorch.\n')
            original_topk_accuracy = original_accuracy_table[model_name]
        else:
            print(f'\nEvaluting the original model to get its accuracy\n')
            original_topk_accuracy = test_accuracy(model, test_loader, topk)
        
        del corrector
        gc.collect()

        print(f'Top-1 accuracy of {model_name} is {original_topk_accuracy[0]}.')
        print(f'Top-5 accuracy of {model_name} is {original_topk_accuracy[1]}.')
        
        start_time = datetime.now()

        print(f'\n Evaluting the quantized model to get its accuracy\n')
        topk_accuracy = test_accuracy(quantized_model, test_loader, topk)
        print(f'Top-1 accuracy of quantized {model_name} is {topk_accuracy[0]}.')
        print(f'Top-5 accuracy of quantized {model_name} is {topk_accuracy[1]}.')

        end_time = datetime.now()

        print(f'\nTime used for evaluation: {end_time - start_time}\n')

        with open(log_file_name, 'a') as f:
            csv_writer = csv.writer(f)
            row = [
                model_name, data_set, batch_size, bc_batch_size,
                original_topk_accuracy[0], topk_accuracy[0], 
                original_topk_accuracy[1], topk_accuracy[1], 
                bits, mlp_scalar, cnn_scalar, 
                mlp_percentile, cnn_percentile, include_0, 
                retain_rate, seed, author
            ]
            csv_writer.writerow(row)
