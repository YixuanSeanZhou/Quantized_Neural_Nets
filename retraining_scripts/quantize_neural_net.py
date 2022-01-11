from __future__ import annotations
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import optim
import numpy as np

import os
import gc
import csv
from tqdm import tqdm

from helper_tools import InterruptException, test_accuracy
from step_algorithm import StepAlgorithm


LINEAR_MODULE_TYPE = nn.Linear
CONV2D_MODULE_TYPE = nn.Conv2d

SUPPORTED_LAYER_TYPE = {LINEAR_MODULE_TYPE, CONV2D_MODULE_TYPE}
SUPPORTED_BLOCK_TYPE = {nn.Sequential}
LAYER_LOG_FILE = '../logs/Layer_Quantize_Log.csv'
fields = [
    'Layer #', 'Layer Type', 'Group', 'Weight Max', 
    'Weight Median', 'Weight Row Max Mean', 
    'Quantization Loss', 'Relative Loss'
]
LAYER_LOGGING_TOGGLE = False
device = "cuda" if torch.cuda.is_available() else "cpu"


class QuantizeNeuralNet():
    '''
    Corresponding object to work with for quantizing the neural network.
    
    Attributes
    ----------
    analog_network : nn.Module
        Copy of the neural network to be quantized.
    batch_size: int
        The batch size to be used to quantize each layer.
    data_loader: function
        The data_loader to load data
    '''
    def __init__(self, network_to_quantize, batch_size, data_loader, 
                retrain_loader, retrain_bs, val_loader,
                 mlp_bits, cnn_bits,
                 include_zero = False, ignore_layers=[], 
                 mlp_alphabet_scalar=1, cnn_alphabet_scalar=1,
                 mlp_percentile=0.5, cnn_percentile=0.5,
                 retain_rate=0.25):
        '''
        Init the object that is used for quantizing the given neural net.
        Parameters
        -----------
        network_to_quantize : nn.Module
            The neural network to be quantized.
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
        self.analog_network = network_to_quantize
        self.batch_size = batch_size
        self.data_loader_iter = iter(data_loader)
        self.retrain_loader = retrain_loader
        self.val_loader = val_loader
        self.retrain_bs = retrain_bs

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
        
        # self.quantized_network = copy.deepcopy(self.analog_network)
        self.analog_network_layers = [] 
        self._extract_layers(self.analog_network, self.analog_network_layers)
        # self.quantized_network_layers = []
        # self._extract_layers(self.quantized_network, self.quantized_network_layers)

    
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

    def quantize_network(self):
        '''
        Perform the quantization of the neural network.
        Parameters
        -----------
        
        Returns
        -------
        nn.Module
            The quantized neural network.
        '''

        layers_to_quantize = [
            i for i, layer in enumerate(self.analog_network_layers) 
                if type(layer) in SUPPORTED_LAYER_TYPE
                    and i not in self.ignore_layers
                ]
        
        print(f'Layer idx to quantize {layers_to_quantize}')
        print(f'Total num to quantize {len(layers_to_quantize)}')

        counter = 0
        
        if LAYER_LOGGING_TOGGLE:
            if os.path.isfile(LAYER_LOG_FILE):
                os.remove(LAYER_LOG_FILE)
            with open(LAYER_LOG_FILE, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
        
        for layer_idx in layers_to_quantize:
  
            gc.collect()

            analog_layer_input = self._populate_linear_layer_input(layer_idx)

            print(f'\nQuantizing layer: {layer_idx}')
            print(f'Quantization progress: {counter} out of {len(layers_to_quantize)}\n')
            counter += 1

            if type(self.analog_network_layers[layer_idx]) == LINEAR_MODULE_TYPE:
                groups = 1
                # Note that each row of W represents a neuron
                W = self.analog_network_layers[layer_idx].weight.data.numpy()

                Q, quantize_error, relative_quantize_error = StepAlgorithm._quantize_layer(W, 
                                                analog_layer_input, 
                                                analog_layer_input,  # quantized input is the same as analog input
                                                analog_layer_input.shape[0],
                                                self.mlp_alphabet * self.mlp_alphabet_scalar, 
                                                self.mlp_percentile
                                                )

                self.analog_network_layers[layer_idx].weight.data = torch.Tensor(Q).float()

            elif type(self.analog_network_layers[layer_idx]) == CONV2D_MODULE_TYPE:

                groups = self.analog_network_layers[layer_idx].groups

                W = self.analog_network_layers[layer_idx].weight.data 
                # W has shape (out_channels, in_channesl/groups, k_size[0], k_size[1])
                W_shape = W.shape
                
                W = W.view(W.size(0), -1).numpy() 
                # shape (out_channels, in_channesl/groups*k_size[0]*k_size[1])
                # each row of W is a neuron (vectorized sliding block)

                Q, quantize_error, relative_quantize_error = StepAlgorithm._quantize_layer(W, 
                                            analog_layer_input, 
                                            analog_layer_input, # quantized input is the same as analog input
                                            analog_layer_input.shape[0],
                                            self.cnn_alphabet * self.cnn_alphabet_scalar,
                                            self.cnn_percentile,
                                            groups=groups
                                            )

                self.analog_network_layers[layer_idx].weight.data = torch.Tensor(Q).float().view(W_shape)
            
            print(f'Shape of weight matrix is {W.shape}')
            print(f'Shape of X is {analog_layer_input.shape}')
            print(f'The quantization error of layer {layer_idx} is {quantize_error}.')
            print(f'The relative quantization error of layer {layer_idx} is {relative_quantize_error}.\n')

            del analog_layer_input
            
            if LAYER_LOGGING_TOGGLE:
                with open(LAYER_LOG_FILE, 'a') as f:
                    csv_writer = csv.writer(f)
                    row = [
                        layer_idx, type(self.analog_network_layers[layer_idx]), groups,
                        np.max(W), np.median(W), np.quantile(np.abs(W), 1, axis=1).mean(),
                        quantize_error, relative_quantize_error
                    ]
                    csv_writer.writerow(row)
            
            gc.collect()

            print(f'\n Evaluting the quantized model.\n')
            topk = (1, 5) 
            topk_accuracy = test_accuracy(self.analog_network, self.val_loader, topk)
            print(f'Top-1 accuracy of quantized model is {topk_accuracy[0]}.')
            print(f'Top-5 accuracy of quantized model is {topk_accuracy[1]}.')

            # freeze the quantized layer
            self.analog_network_layers[layer_idx].weight.requires_grad = False
            self.analog_network_layers[layer_idx].bias.requires_grad = False
            
            # re-training the neural network 
            if counter > 10:
                n_epochs = 1
                lr = 1 * 1e-4
                weight_decay = 1e-6 
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.analog_network.parameters()), 
                                    lr=lr, weight_decay=weight_decay)
                self.analog_network.train()
                print(f'\n Start re-training: {device} is available, num_epochs {n_epochs}, \
                            batch size {self.retrain_bs}, lr {lr}\n')

                for epoch in range(1, n_epochs+1):
                    batch_losses = []
                    for x_batch, y_batch in tqdm(self.retrain_loader):
                        output = self.analog_network(x_batch)
                        loss = F.cross_entropy(output, y_batch)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        batch_losses.append(loss.item())
                    training_loss = np.mean(batch_losses)
                    if (epoch <= 20) | (epoch % 10 == 0):
                        print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}")
                self.analog_network.eval()  

                print(f'\n Evaluting the retrained model.\n')
                topk = (1, 5) 
                topk_accuracy = test_accuracy(self.analog_network, self.val_loader, topk)
                print(f'Top-1 accuracy of quantized model is {topk_accuracy[0]}.')
                print(f'Top-5 accuracy of quantized model is {topk_accuracy[1]}.')
        
        return self.analog_network

        
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
        # quantized_handle = self.quantized_network_layers[layer_idx].register_forward_hook(save_input)
        
        with torch.no_grad():
            try:
                self.analog_network(raw_input_data)
            except InterruptException:
                pass

            analog_handle.remove()

            # try:
            #     self.quantized_network(raw_input_data)
            # except InterruptException:
            #     pass

            # quantized_handle.remove()
        
        del raw_input_data
        gc.collect()

        return save_input.inputs[0]


class SaveInputMLP:
    """
    This class is used to store inputs from original/quantized neural networks
    """
    def __init__(self):
        self.inputs = []
        # self.batch_size = batch_size
        # self.p = 0.125
        # self.call_count = 0
        # self.rand_indices = []

    def __call__(self, module, module_in, module_out):
        if len(module_in) != 1:
            raise TypeError('The number of input layer is not equal to one!')
    
        self.inputs.append(module_in[0].numpy())
        raise InterruptException

        # batch_size = module_in[0].shape[0]

        # if self.call_count == 0:
        #     self.rand_indices = np.random.choice(np.arange(0, batch_size), size=int(self.p*batch_size + 1 if self.p != 1 else batch_size)) 
        # self.call_count += 1
        
        # module_in_numpy = module_in[0].numpy()

        # self.inputs.append(module_in_numpy[self.rand_indices])
        
        # raise InterruptException


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
        
        