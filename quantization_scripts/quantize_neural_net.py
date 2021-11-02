from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from helper_tools import InterruptException
from step_algorithm import StepAlgorithm

TEMP_ANALOG_TENSOR_FILE = 'temp_input_tensor_analog_file.pt'
TEMP_QUANTIZED_TENSOR_FILE = 'temp_input_tensor_quantized_file.pt'

LINEAR_MODULE_TYPE = nn.Linear
CONV2D_MODULE_TYPE = nn.Conv2d

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
    def __init__(self, network_to_quantize, batch_size, data_loader, bits,
                 ignore_layers=[], alphabet_scalar=1):
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
        bits : int
            Num of bits that alphabet is used.
        ignore_layers : List[int]
            List of layer index that shouldn't be quantized.
        alphabet_scaler: float,
            The alphabet_scaler used to determine the radius \
            of the alphabet for each layer.
        
        Returns
        -------
        QuantizeNeuralNet
            The object that is used to perform quantization
        '''
        self.analog_network = network_to_quantize
        self.batch_size = batch_size
        self.data_loader_iter = iter(data_loader)

        # FIXME: alphabet_scaler should probably not be used like this
        self.alphabet_scalar = alphabet_scalar
        self.bits = bits
        self.alphabet = np.linspace(-1, 1, num=int(2 ** bits))
        np.append(self.alphabet, 0)
        # self.alphabet = np.array([0, -1, 1])
        self.ignore_layers = ignore_layers

        # create a copy which is our quantized network
        # self.quantized_network = type(self.analog_network)(
        #     self.analog_network.input_dim, 
        #     self.analog_network.hidden_dim, 
        #     self.analog_network.outputdim
        #     )
        # self.quantized_network.load_state_dict(self.analog_network.state_dict())
        self.quantized_network = copy.deepcopy(self.analog_network)
        # self.quantized_network.load_state_dict(self.analog_network.state_dict())
        # print(type(self.quantized_network))

        self.analog_network_layers = [] 
        self._extract_layers(self.analog_network, self.analog_network_layers)
        self.quantized_network_layers = []
        self._extract_layers(self.quantized_network, self.quantized_network_layers)

    
    def _extract_layers(self, network, layer_list):
        """
        Recursively obtain layers of given network
        """
        for layer in network.children():
            if type(layer) == nn.Sequential:
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
            i for i, layer in enumerate(self.quantized_network.children()) 
                if type(layer) == LINEAR_MODULE_TYPE or CONV2D_MODULE_TYPE
                    and i not in self.ignore_layers
                ]

        for layer_idx in layers_to_quantize:
            analog_layer_input, quantized_layer_input \
                = self._populate_linear_layer_input(layer_idx)

            if type(self.analog_network_layers[layer_idx]) == LINEAR_MODULE_TYPE:

                # Note that each row of W represents a neuron
                W = self.analog_network_layers[layer_idx].weight.data.numpy()

                Q, quantize_error, relative_quantize_error = StepAlgorithm._quantize_layer(W, 
                                                analog_layer_input, 
                                                quantized_layer_input, 
                                                analog_layer_input.shape[0],
                                                self.alphabet * self.alphabet_scalar
                                                )

                self.quantized_network_layers[layer_idx].weight.data = torch.Tensor(Q).float()

            elif type(self.analog_network_layers[layer_idx]) == CONV2D_MODULE_TYPE:

                # Note that each row of W represents a neuron
                W = self.analog_network_layers[layer_idx].weight.data.numpy()

                W_shape = W.shape

                kernel_size = self.analog_network_layers[layer_idx].kernel_size

                W = W.reshape(-1, kernel_size[0] * kernel_size[1])

                Q, quantize_error, relative_quantize_error = StepAlgorithm._quantize_layer(W, 
                                            analog_layer_input, 
                                            quantized_layer_input, 
                                            analog_layer_input.shape[0],
                                            self.alphabet * self.alphabet_scalar
                                            )
                
                Q = Q.reshape(W_shape)

                self.quantized_network_layers[layer_idx].weight.data = torch.Tensor(Q).float()

            print(f'The quantization error of layer {layer_idx} is {quantize_error}.')
            print(f'The relative quantization error of layer {layer_idx} is {relative_quantize_error}.')
            print()

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
                stride=analog_layer.stride
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
        return (save_input.inputs[0], save_input.inputs[1])

    

class SaveInputMLP:
    """
    This class is used to store inputs from original/quantized neural networks
    """
    def __init__(self):
        self.inputs = []
        

    def __call__(self, module, module_in, module_out):
        if len(module_in) != 1:
            raise TypeError('The number of input layer is not equal to one!')
        self.inputs.append(module_in[0].numpy())
        raise InterruptException


class SaveInputConv2d:
    '''
    This class is useed to store inputs from original/quantizeed neural netwroks
    for conv layers.
    '''
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
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
        '''
        self.kernel_size_2d = kernel_size[0] * kernel_size[1]
        self.unfolder = torch.nn.Unfold(kernel_size, dilation, padding, stride)
        self.inputs = []
        

    def __call__(self, module, module_in, module_out):
        if len(module_in) != 1:
            raise TypeError('The number of input layer is not equal to one!')
        unfolded = self.unfolder(module_in[0])
        unfolded = torch.transpose(unfolded, -1, -2)
        # FIXME: this might need revisit
        unfolded = unfolded.reshape(-1, self.kernel_size_2d)
        self.inputs.append(unfolded.numpy())
        raise InterruptException
    

    def _unfold_conv2d_input(self, module_in):
        '''
        Convert conv2d input layer to sliding window.

        Parameters
        -----------
        module_in : torch.tensor
            The input feature map to the conv2d layer of the analog network
        
        Returns
        -------
        torch.tensor
            A tensor of sliding input results. A long the column, it conatins
            the windows of each sample in the batch stacked together.
        '''
        pass


