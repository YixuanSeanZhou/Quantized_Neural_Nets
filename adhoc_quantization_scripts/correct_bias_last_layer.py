from __future__ import annotations
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.efficientnet import ConvNormActivation, SqueezeExcitation, MBConv 
from torchvision.models.mobilenetv2 import InvertedResidual
import torch.nn.functional as F
import numpy as np
import copy

import os
import gc
import csv

from helper_tools import InterruptException
from step_algorithm import StepAlgorithm


LINEAR_MODULE_TYPE = nn.Linear
CONV2D_MODULE_TYPE = nn.Conv2d

SUPPORTED_LAYER_TYPE = {LINEAR_MODULE_TYPE, CONV2D_MODULE_TYPE}
SUPPORTED_BLOCK_TYPE = {nn.Sequential, 
                        BasicBlock, Bottleneck,
                        BasicConv2d, Inception, InceptionAux,
                        ConvNormActivation, SqueezeExcitation, MBConv,
                        InvertedResidual
                        }

class CorrectBiasLastLayer():
    '''
    Corresponding object to restore the last layer and bias correct it.
    
    Attributes
    ----------
    analog_network : nn.Module
        Copy of the neural network that is already quantized.
    batch_size: int
        The batch size to be used to perform bias correction.
    data_loader: function
        The data_loader to load data
    '''
    def __init__(self, analog_network, quantized_network, 
                 batch_size, data_loader, 
                 mlp_bits, cnn_bits,
                 include_zero = False, ignore_layers=[], 
                 mlp_alphabet_scalar=1, cnn_alphabet_scalar=1,
                 mlp_percentile=0.5, cnn_percentile=0.5,
                 retain_rate=0.25):
        '''
        Init the object that is used for quantizing the given neural net.
        Parameters
        -----------
        analog_network : nn.Module
            The neural network to be quantized.
        quantized_network : nn.Module
            The quantized neural network.
        batch_size: int,
            The batch size input to each layer when quantization is performed.
        data_loader: function,
            The generator that loads the raw dataset
        mlp_bits : int
            Num of bits that mlp alphabet is used.
        cnn_bits: int
            Num of bits that cnn alphabet is used.
        include_zero: bool
            Indicate whether to augment the alphabet with a 0.
        ignore_layers : List[int]
            List of layer index that shouldn't be quantized.
        mlp_alphabet_scaler: float,
            The alphabet_scaler used to determine the radius \
            of the alphabet for each mlp layer.
        cnn_alphabet_scaler: float,
            The alphabet_scaler used to determine the radius \
            of the alphabet for each cnn layer.
        mlp_percentile: float,
            The percentile to use for finding each mlp layer's alphabet.
        cnn_percentile: float,
            The percentile to use for finding each cnn layer's alphabet.
        retain_rate: float:
            The ratio to retain after unfold.
        Returns
        -------
        QuantizeNeuralNet
            The object that is used to perform quantization
        '''
        self.analog_network = analog_network
        self.quantized_network = quantized_network
        self.batch_size = batch_size
        self.data_loader_iter = iter(data_loader)

        self.mlp_alphabet_scalar = mlp_alphabet_scalar
        self.cnn_alphabet_scalar = cnn_alphabet_scalar

        self.mlp_bits = mlp_bits
        self.cnn_bits = cnn_bits
        self.mlp_alphabet = np.linspace(-1, 1, num=int(2 ** mlp_bits))
        self.cnn_alphabet = np.linspace(-1, 1, num=int(2 ** cnn_bits))
        
        if include_zero:
            self.mlp_alphabet = np.append(self.mlp_alphabet, 0)
            self.cnn_alphabet = np.append(self.cnn_alphabet, 0)
        
        self.mlp_percentile = mlp_percentile
        self.cnn_percentile = cnn_percentile

        self.ignore_layers = ignore_layers
        self.retain_rate = retain_rate
        
        self.analog_network_layers = [] 
        self._extract_layers(self.analog_network, self.analog_network_layers)
        self.quantized_network_layers = []
        self._extract_layers(self.quantized_network, self.quantized_network_layers)

    
    def _extract_layers(self, network, layer_list):
        """
        Recursively obtain layers of given network
        """
        for layer in network.children():
            if type(layer) in SUPPORTED_BLOCK_TYPE:
                # if sequential layer, apply recursively to layers in sequential layer
                self._extract_layers(layer, layer_list)
            if not list(layer.children()):
                # if leaf node, add it to list
                layer_list.append(layer)

    def correct_bias_for_last_layer_network(self):
        '''
        Replace the last layer to be unquantized version, 
        bias correct the last layer using expected outputs.
        
        Parameters
        -----------
        Returns
        -------
        nn.Module
            The quantized neural network.
        '''

        layers_to_quantize = [
            i for i, layer in enumerate(self.quantized_network_layers) 
                if type(layer) in SUPPORTED_LAYER_TYPE
                    and i not in self.ignore_layers
                ]
        
        print(f'Layer idx to quantize {layers_to_quantize}')
        print(f'Total num to quantize {len(layers_to_quantize)}')

        last_layer_idx = layers_to_quantize[-1]

        if type(self.analog_network_layers[last_layer_idx]) != LINEAR_MODULE_TYPE:
            print(f'\033[91m Last layer is not MLP, failed to correct \033[0m')
        
        else:
            # replace the last layer to be unquantized version
            self.quantized_network_layers[last_layer_idx].weight.data = self.analog_network_layers[last_layer_idx].weight.data

            W = self.analog_network_layers[last_layer_idx].weight.data.numpy()           
            Q = self.quantized_network_layers[last_layer_idx].weight.data.numpy()
            if self.analog_network_layers[last_layer_idx].bias is None:
                print('f''\033[91m Last layer has no bias, failed to correct \033[0m')
                return self.quantized_network
            b = self.analog_network_layers[last_layer_idx].bias.data.numpy()

            analog_layer_input, quantized_layer_input \
                = self._populate_linear_layer_input(last_layer_idx)
            
            b_prime = StepAlgorithm.bias_correction(
                                      analog_layer_input, 
                                      quantized_layer_input, 
                                      W, Q, b, analog_layer_input.shape[0])
            
            self.quantized_network_layers[last_layer_idx].bias.data = torch.Tensor(b_prime).float()

        del analog_layer_input, quantized_layer_input

        gc.collect()

        return self.quantized_network

        
    def _populate_linear_layer_input(self, layer_idx):
        '''
        Load the input to the given layer specified by the layer_idx for both
        analog network and the network to be quantized.
        Parameters
        -----------
        layer_idx : int
            The idx of the layer to be quantized.
        Returns
        -------
        tuple(torch.Tensor)
            A tuple of torch.Tensor that is the input for the intersted layer, 
            at 0th idx is the input for the analog network layer,
            at 1st idx is the input for the quantizing network layer.
        '''

        # get data
        raw_input_data, _ = next(self.data_loader_iter)

        analog_layer = self.analog_network_layers[layer_idx]

        # save_input instances
        if type(analog_layer) == LINEAR_MODULE_TYPE:
            save_input = SaveInputMLP()
        elif type(analog_layer) == CONV2D_MODULE_TYPE:
            save_input = SaveInputConv2d(
                kernel_size=analog_layer.kernel_size,
                dilation=analog_layer.dilation,
                padding=analog_layer.padding,
                stride=analog_layer.stride,
                groups=analog_layer.groups,
                retain_rate=self.retain_rate
            )
        else:
            raise TypeError(f'The layer type {type(analog_layer)} is not currently supported')

        # attach handles to both analog and quantize layer
        analog_handle = self.analog_network_layers[layer_idx].register_forward_hook(save_input)
        quantized_handle = self.quantized_network_layers[layer_idx].register_forward_hook(save_input)
        
        with torch.no_grad():
            try:
                self.analog_network(raw_input_data)
            except InterruptException:
                pass

            analog_handle.remove()

            try:
                self.quantized_network(raw_input_data)
            except InterruptException:
                pass

            quantized_handle.remove()
        
        del raw_input_data
        gc.collect()

        return (save_input.inputs[0], save_input.inputs[1])


class SaveInputMLP:
    """
    This class is used to store inputs from original/quantized neural networks
    """
    def __init__(self):
        self.inputs = []

    def __call__(self, module, module_in, module_out):
        '''
        Process the input to the attached layer and save in self.inputs
        '''
        if len(module_in) != 1:
            raise TypeError('The number of input layer is not equal to one!')
    
        self.inputs.append(module_in[0].numpy())
        raise InterruptException


class SaveInputConv2d:
    '''
    This class is useed to store inputs from original/quantizeed neural netwroks
    for conv layers.
    '''
    def __init__(self, kernel_size, dilation, padding, stride, groups, retain_rate):
        '''
        Init the SaveInputConv2d object
        
        Parameters:
        -----------
        kernal_size: int or tuple
            The size of the layer's kernel
        dilation: int or tuple
            The dialation of the conv2d layer who takes the input
        padding: iint or tuple
            The padding used of the conv2d layer who takes the input
        stride: int or tuple
            The stride of of the conv2d layer who takes the input
        groups: int
            The groups in the conv2d layer
        retain_rate: float
            The ratio to retain after unfold.
        '''
        self.p = retain_rate
        self.unfolder = nn.Unfold(kernel_size, dilation, padding, kernel_size)
        self.inputs = []
        self.call_count = 0
        self.groups = groups


    def __call__(self, module, module_in, module_out):
        '''
        Process the input to the attached layer and save in self.inputs
        '''
        if len(module_in) != 1:
            raise TypeError('The number of input layer is not equal to one!')
        # module_in has shape (B, C, H, W)
        module_input = module_in[0]

        # Need to consider both batch_size B and in_channels C
        unfolded = self.unfolder(module_input)  # shape (B, C*kernel_size[0]*kernel_size[1], L)
        
        batch_size, num_blocks = unfolded.shape[0], unfolded.shape[-1]
        unfolded = torch.transpose(unfolded, 1, 2) # shape (B, L, C*kernel_size[0]*kernel_size[1])
        unfolded = unfolded.reshape(-1, unfolded.size(-1)).numpy() # shape (B*L, C*kernel_size[0]*kernel_size[1])

        if self.call_count == 0:
            self.rand_indices = np.concatenate(
                        [np.random.choice(np.arange(num_blocks*i, num_blocks*(i+1)), 
                        size=int(self.p*num_blocks + 1 if self.p != 1 else self.p*num_blocks)) 
                        for i in range(batch_size)]
                        ) # need to define self.p (probability)
        self.call_count += 1
        unfolded = unfolded[self.rand_indices]

        self.inputs.append(unfolded)
        raise InterruptException