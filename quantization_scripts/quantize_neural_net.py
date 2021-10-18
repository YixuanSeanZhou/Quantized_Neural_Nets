import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import numpy

from .helper_tools import InterruptException
from .step_algorithm import StepAlgorithm

from typing import List, final

TEMP_ANALOG_TENSOR_FILE = 'temp_input_tensor_analog_file.pt'
TEMP_QUANTIZED_TENSOR_FILE = 'temp_input_tensor_quantized_file.pt'

LINEAR_MODULE_TYPE = torch.nn.modules.linear.Linear

class QuantizeNeuralNet():
    def __init__(self, 
                 network_to_quantize: torch.nn.Module, 
                 batch_size: int,
                 data_loader: function,
                 mini_batch_size: int,
                 bits: float,
                 ignore_layers: List[int] =[],
                 alphabet_scalar: float = 1):
        '''
        Init the object that is used for quantizing the given neural net.
        '''
        self.analog_network = network_to_quantize
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.min_batch_size = mini_batch_size

        # FIXME: alphabet_scaler should probably not be used like this
        self.alphabet_scalar = alphabet_scalar
        self.bits = bits
        self.alphabet = np.linspace(-1, 1, num=int(round(2 ** (bits))))

        self.ignore_layers = ignore_layers

        # create a copy which is our quantized network
        self.quantized_network = type(self.analog_network)()
        self.quantized_network.load_state_dict(self.analog_network.state_dict())

        self.analog_network_layers = list(self.analog_network.children())
        self.quantized_network_layers = list(self.quantized_network.children())

    def quantize_network(self):
        layers_to_quantize = [
            i for i, layer in enumerate(self.quantized_network.children()) 
                if layer.type() == LINEAR_MODULE_TYPE 
                    and i not in self.ignore_layers
                ]

        for layer_idx in layers_to_quantize:
            analog_layer_input, quantized_layer_input \
                = self._populate_linear_layer_input(layer_idx)

            W = self.analog_network_layers[layer_idx].weight.data.detach().numpy()

            layer_alphabet = numpy.mean(W.flatten()) * self.alphabet

            Q = StepAlgorithm._quantize_layer(W, 
                                              analog_layer_input, 
                                              quantized_layer_input, 
                                              self.batch_size,
                                              layer_alphabet
                                              )

            quantized_layer_input[layer_idx].weight.data = Q

        
    def _populate_linear_layer_input(self, layer_idx: int):
        # get data
        raw_input_data = next(iter(self.data_loader))

        # attach a handle to both analog and quantize layer
        analog_handle = self.analog_network_layers[layer_idx]\
                            .register_forward_hook(
                                QuantizeNeuralNet._layer_input_extract_hook
                                )
        quantized_handle = self.quantized_network_layers[layer_idx]\
                            .register_forward_hook(
                                QuantizeNeuralNet._layer_input_extract_hook
                                )
        
        # make evaluation for both models
        torch.no_grad()
        
        try:
            self.analog_model(raw_input_data)
        except InterruptException:
            pass
        
        analog_handle.remove()

        try:
            self.quantized_model(raw_input_data)
        except InterruptException:
            pass

        quantized_handle.remove()
        
        torch.enable_grad()

        analog_input = torch.load(TEMP_ANALOG_TENSOR_FILE).detach().numpy()
        quantized_input = torch.load(TEMP_QUANTIZED_TENSOR_FILE).detach().numpy()
        
        return (analog_input, quantized_input)

    def _analog_layer_input_extract_hook(module, layer_input, layer_output):
        '''
        The forward hook used to extract the layer's input of a module,
        InterruptException is thrown to terminate the forward evaluation.
        '''
        if os.path.exists(TEMP_ANALOG_TENSOR_FILE):
            os.remove(TEMP_ANALOG_TENSOR_FILE)
        if len(layer_input) != 1:
            raise TypeError('The num input layer is not one')
        torch.save(layer_input[0], TEMP_ANALOG_TENSOR_FILE)
        raise InterruptException
    
    def _quantized_layer_input_extract_hook(module, layer_input, layer_output):
        '''
        The forward hook used to extract the layer's input of a module,
        InterruptException is thrown to terminate the forward evaluation.
        '''
        if os.path.exists(TEMP_QUANTIZED_TENSOR_FILE):
            os.remove(TEMP_QUANTIZED_TENSOR_FILE)
        if len(layer_input) != 1:
            raise TypeError('The num input layer is not one')
        torch.save(layer_input[0], TEMP_QUANTIZED_TENSOR_FILE)
        raise InterruptException

