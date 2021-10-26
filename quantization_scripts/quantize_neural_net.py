from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import numpy
import copy

from helper_tools import InterruptException
from step_algorithm import StepAlgorithm

TEMP_ANALOG_TENSOR_FILE = 'temp_input_tensor_analog_file.pt'
TEMP_QUANTIZED_TENSOR_FILE = 'temp_input_tensor_quantized_file.pt'

LINEAR_MODULE_TYPE = nn.Linear

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

        self.analog_network_layers = list(self.analog_network.children())
        self.quantized_network_layers = list(self.quantized_network.children())

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
                if type(layer) == LINEAR_MODULE_TYPE 
                    and i not in self.ignore_layers
                ]

        for layer_idx in layers_to_quantize:
            analog_layer_input, quantized_layer_input \
                = self._populate_linear_layer_input(layer_idx)

            W = self.analog_network_layers[layer_idx].weight.data.numpy()

            layer_alphabet \
                = numpy.mean(W.flatten()) * self.alphabet_scalar * self.alphabet

            Q = StepAlgorithm._quantize_layer(W, 
                                              analog_layer_input, 
                                              quantized_layer_input, 
                                              self.batch_size,
                                              layer_alphabet
                                              )

            self.quantized_network_layers[layer_idx].weight.data = torch.tensor(Q).float()

            print(f'Finished quantizing layer {layer_idx}')

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

        # save_input instances
        save_input = SaveInput()

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
    

class SaveInput:
    """
    This class is used to store inputs from original/quantized neural networks
    """
    def __init__(self):
        self.inputs = []
        

    def __call__(self, module, module_in, module_out):
        if len(module_in) != 1:
            raise TypeError('The number of input layer is not equal to one!')
        self.inputs.append(module_in[0])
        raise InterruptException
        