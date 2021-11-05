from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
import os

from quantize_neural_net import QuantizeNeuralNet
from train_conv2d import test
from data_loaders import data_loader, data_loader_miniimagenet

from models import LeNet5, CNN

def augment(x):
    return x.repeat(3, 1, 1)

if __name__ == '__main__':
    batch_size = 32  # batch_size used for quantization
    num_workers = 4
    bits = 1
    default_transform = transforms.ToTensor()
    LeNet_transform = transforms.Compose([transforms.Resize((32, 32)), 
                                    transforms.ToTensor()])
    AlexTransform = transforms.Compose([
                    transforms.Resize((63, 63)),
                    transforms.ToTensor(),
                    augment
                    ])
    min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    # see https://pytorch.org/vision/stable/models.html
    transform_pipeline = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Resize(min_img_size),   
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                         ])
    # transform_pipeline is used for all pretrained models and Normalize is mandatory

    transform = default_transform
    model_name = 'vgg16'   # choose models trained by ourselves 
    model_path = os.path.join('../models', model_name) # only needed for our trained models
    ds_name = 'MiniImagenet'    # name of dataset, use names in following link 
    # https://pytorch.org/vision/stable/datasets.html#fashion-mnist

    # load the model to be quantized
    if model_name in ['vgg16', 'vgg16_bn']:  # add more models later
        model = getattr(torchvision.models, model_name)(pretrained=True) 
        model.eval()  # eval() is necessary 
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        # choose dataset here
    
    if ds_name == 'MiniImagenet':
        train_loader, _, test_loader = data_loader_miniimagenet(batch_size, transform, num_workers)
    else:
    # load the data loader for training and testing
        train_loader, _, test_loader = data_loader(ds_name, batch_size, train_ratio=1, 
                                      num_workers=num_workers, 
                                      transform=transform
                                        )
    
    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, batch_size, train_loader, bits=bits)
    quantized_model = quantizer.quantize_network()
    predictions, labels = test(test_loader, quantized_model)
    test_accuracy = np.sum(predictions == labels) / len(labels)
    torch.save(quantized_model, f'../models/quantized_b{bits}_'+model_name)
    print(f'The testing accuracy is: {test_accuracy}.')
