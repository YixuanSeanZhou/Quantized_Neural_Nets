import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import copy
import gc
import os

from utils import InterruptException, extract_layers
from step_algorithm import StepAlgorithm

LINEAR_MODULE_TYPE = nn.Linear
CONV2D_MODULE_TYPE = nn.Conv2d

RESULT_LOGGING_DIR = 'result_logging'
LAYER_LOGGING = False

class QuantizeNeuralNet:
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
    def __init__(self,
                 network_to_quantize, network_name, batch_size, data_loader, 
                 mlp_bits, cnn_bits,
                 ignore_layers, 
                 mlp_alphabet_scalar, cnn_alphabet_scalar,
                 mlp_percentile, cnn_percentile,
                 reg, lamb, retain_rate, stochastic_quantization, device):
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
        ignore_layers : List[int]
            List of layer index that shouldn't be quantized.
        mlp_alphabet_scaler: float,
            The alphabet_scaler used to determine the radius 
            of the alphabet for each mlp layer.
        cnn_alphabet_scaler: float,
            The alphabet_scaler used to determine the radius 
            of the alphabet for each cnn layer.
        mlp_percentile: float,
            The percentile to use for finding each mlp layer's alphabet.
        cnn_percentile: float,
            The percentile to use for finding each cnn layer's alphabet.
        reg: str
            The type of regularizer to be used.
        lamb: float
            The lambda for regularization.
        retain_rate: float:
            The ratio to retain after unfold.
        stochastic_quantization: bool
            Whether to use stochastic quantization or not
        device: torch.device
            CUDA or CPU
    
        Returns
        -------
        QuantizeNeuralNet
            The object that is used to perform quantization
        '''
        self.network_name = network_name
        self.analog_network = network_to_quantize
        self.batch_size = batch_size
        self.data_loader_iter = iter(data_loader)

        # determine the boundary idx which equals to 2^(bits - 1) for symmetry
        self.mlp_boundary_idx = 2 ** (mlp_bits - 1)
        self.cnn_boundary_idx = 2 ** (cnn_bits - 1)
        
        # rescale the step_size by scalar
        self.mlp_alphabet_scalar = mlp_alphabet_scalar
        self.mlp_alphabet_step_size = mlp_alphabet_scalar / self.mlp_boundary_idx
        self.cnn_alphabet_step_size = cnn_alphabet_scalar / self.cnn_boundary_idx

        self.mlp_bits = mlp_bits
        self.cnn_bits = cnn_bits

        self.mlp_percentile = mlp_percentile
        self.cnn_percentile = cnn_percentile

        self.ignore_layers = ignore_layers
        self.retain_rate = retain_rate
        
        self.reg = reg
        self.lamb = lamb
        self.device = device
        
        self.quantized_network = copy.deepcopy(self.analog_network)
        self.analog_network_layers = [] 
        extract_layers(self.analog_network, self.analog_network_layers)
        self.quantized_network_layers = []
        extract_layers(self.quantized_network, self.quantized_network_layers)
        
        self.stochastic_quantization = stochastic_quantization


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
        layers_to_quantize = [i for i in range(len(self.quantized_network_layers))
                    if i not in self.ignore_layers]

        print(f'Layer indices to quantize {layers_to_quantize}')
        print(f'Total number of layers to quantize {len(layers_to_quantize)}')

        counter = 0
        
        for layer_idx in layers_to_quantize:
            gc.collect()
            analog_layer_input, quantized_layer_input = self._populate_linear_layer_input(layer_idx)

            print(f'\nQuantizing layer with index: {layer_idx}')
            print(f'Quantization progress: {counter} out of {len(layers_to_quantize)-1}\n')
            counter += 1

            if type(self.analog_network_layers[layer_idx]) == LINEAR_MODULE_TYPE:

                groups = 1
                # Note that each row of W represents a neuron
                W = self.analog_network_layers[layer_idx].weight.data

                Q, quantize_error, relative_quantize_error, quantize_adder, relative_adder = StepAlgorithm._quantize_layer(
                                                W, 
                                                analog_layer_input, 
                                                quantized_layer_input, 
                                                analog_layer_input.shape[0],
                                                self.mlp_alphabet_step_size,
                                                self.mlp_boundary_idx,
                                                self.mlp_percentile,
                                                self.reg, self.lamb,
                                                groups, self.stochastic_quantization,
                                                self.device
                                                )

                self.quantized_network_layers[layer_idx].weight.data = Q.float()

            elif type(self.analog_network_layers[layer_idx]) == CONV2D_MODULE_TYPE:
                groups = self.analog_network_layers[layer_idx].groups

                W = self.analog_network_layers[layer_idx].weight.data 
                # W has shape (out_channels, in_channesl/groups, k_size[0], k_size[1])

                print('shape of W:', W.shape)
                print('shape of analog_layer_input:', analog_layer_input.shape)
                print('shape of quantized_layer_input:', quantized_layer_input.shape)
                
                W_shape = W.shape
                W = W.view(W.size(0), -1)
                # shape (out_channels, in_channesl/groups*k_size[0]*k_size[1])
                # each row of W is a neuron (vectorized sliding block)

                Q, quantize_error, relative_quantize_error, quantize_adder, relative_adder = StepAlgorithm._quantize_layer(
                                            W, 
                                            analog_layer_input, 
                                            quantized_layer_input, 
                                            analog_layer_input.shape[0],
                                            self.cnn_alphabet_step_size,
                                            self.cnn_boundary_idx,
                                            self.cnn_percentile,
                                            self.reg, self.lamb,
                                            groups, self.stochastic_quantization,
                                            self.device
                                            )
                
                self.quantized_network_layers[layer_idx].weight.data = Q.reshape(W_shape).float()
            
            print(f'The quantization error of layer {layer_idx} is {quantize_error.cpu().numpy()}.')
            print(f'The relative quantization error of layer {layer_idx} is {relative_quantize_error.cpu().numpy()}.\n')

            # layer logging
            if LAYER_LOGGING:
                model_name = self.network_name
                bits = self.mlp_bits
                ori_layer_weight_file_name = f'batch_size:{self.batch_size}_model_name:{model_name}_bits:{bits}_scalar:{self.mlp_alphabet_scalar}_stochastic:{self.stochastic_quantization}_layer:{layer_idx}_ori.npy'
                np.save(os.path.join(RESULT_LOGGING_DIR, ori_layer_weight_file_name), W)
                quant_layer_weight_file_name = f'batch_size:{self.batch_size}_model_name:{model_name}_bits:{bits}_scalar:{self.mlp_alphabet_scalar}_stochastic:{self.stochastic_quantization}_layer:{layer_idx}_quant.npy'
                np.save(os.path.join(RESULT_LOGGING_DIR, quant_layer_weight_file_name), Q)
                quant_layer_adder_file_name = f'batch_size:{self.batch_size}_model_name:{model_name}_bits:{bits}_scalar:{self.mlp_alphabet_scalar}_stochastic:{self.stochastic_quantization}_layer:{layer_idx}_adder.npy'
                np.save(os.path.join(RESULT_LOGGING_DIR, quant_layer_adder_file_name), quantize_adder)
                quant_layer_relative_file_name = f'batch_size:{self.batch_size}_model_name:{model_name}_bits:{bits}_scalar:{self.mlp_alphabet_scalar}_stochastic:{self.stochastic_quantization}_layer:{layer_idx}_relative.npy'
                np.save(os.path.join(RESULT_LOGGING_DIR, quant_layer_relative_file_name), relative_adder)

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
                self.analog_network(raw_input_data.to(self.device))
            except InterruptException:
                pass

            analog_handle.remove()

            try:
                self.quantized_network(raw_input_data.to(self.device))
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
    
        self.inputs.append(module_in[0])
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
        # module_input has shape (B, C, H, W)
        module_input = module_in[0]
        # Need to consider both batch_size B and in_channels C
        unfolded = self.unfolder(module_input)  # shape (B, C*kernel_size[0]*kernel_size[1], L)
        
        batch_size, num_blocks = unfolded.shape[0], unfolded.shape[-1]
        unfolded = torch.transpose(unfolded, 1, 2) # shape (B, L, C*kernel_size[0]*kernel_size[1])
        unfolded = unfolded.reshape(-1, unfolded.size(-1)) # shape (B*L, C*kernel_size[0]*kernel_size[1])

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
