import torch
import torch.nn.functional as F
from tqdm import tqdm

class StepAlgorithm:
    
    def _stochastic_msq(step_size, x, boundary_idx, lamb):
        '''
        First version of stochastic msq without regularizer.
        The quantization result is clipped to be within boundary idx.
        
        Parameters
        ----------
        step_size: float
            The step size of the alphabet
        x: float
            The value to be quantized
        boundary_idx: int
            The max idx of the alphebt to not go over
        lamb: dummy variable, not used
        
        Returns
        -------
        float of the result of msq
        '''
        # stochastic quantization
        p = 1 - x / step_size + torch.floor(x / step_size)  # probability
        prob_mask = torch.bernoulli(p).bool()  
        x[prob_mask] = step_size * torch.floor(x[prob_mask] / step_size) 
        x[~prob_mask] = step_size * (torch.floor(x[~prob_mask] / step_size) + 1)

        # clipping large values
        clipping_mask = (torch.abs(x) > step_size * boundary_idx)
        x[clipping_mask] = torch.sign(x[clipping_mask]) * step_size * boundary_idx
        return x
    
    
    def _msq(step_size, x, boundary_idx, lamb):
        '''
        Assuming the alphebt is uniform and symmetric, perform msq
        
        Parameters
        ----------
        step_size: float
            The step size of the alphabet
        x: float
            The value to be quantized
        boundary_idx: int
            The max idx of the alphebt to not go over
        lamb: dummy variable, not used
        
        Returns
        -------
        float of the result of msq
        '''
        return torch.sign(x) * step_size * torch.minimum(torch.abs(torch.floor(x / step_size + 0.5)), torch.ones_like(x) * boundary_idx)

    
    def _hard_thresholding_msq(step_size, x, boundary_idx, lamb):
        '''
        Hard thresholding quantizer.
        
        Parameters
        ----------
        step_size: float
            The step size of the alphabet
        x: float
            The value to be quantized
        boundary_idx: int
            The max idx of the alphebt to not go over
        lamb: float
            The boundary for threasholding
        
        Returns
        -------
        Floating value result of hard thresholding
        '''
        x = F.threshold(torch.abs(x), lamb, 0) * torch.sign(x)  # hard thresholding 
        y = torch.sign(x) * torch.maximum(torch.abs(x)-lamb, torch.zeros_like(x))  # soft thresholding 
        round_val = torch.minimum(torch.abs(torch.floor(y / step_size + 0.5)), torch.ones_like(y) * boundary_idx)
        return torch.sign(x) * (lamb + step_size * round_val) * (torch.abs(x) > lamb).float()


    def _soft_thresholding_msq(step_size, x, boundary_idx, lamb):
        '''
        Soft thresholding quantizer.
        
        Parameters
        ----------
        step_size: float
            The step size of the alphabet
        x: float
            The value to be quantized
        boundary_idx: int
            The max idx of the alphebt to not go over
        lamb: float
            The boundary for threasholding
        
        Returns
        -------
        Floating value result of hard thresholding
        '''
        x = torch.sign(x) * torch.maximum(torch.abs(x)-lamb, torch.zeros_like(x))  # soft thresholding 
        return torch.sign(x) * step_size * torch.minimum(torch.abs(torch.floor(x / step_size + 0.5)), torch.ones_like(x) * boundary_idx)
        
    
    def _quantization(W, Q, U, analog_layer_input, quantized_layer_input, quantizer, 
                      step_size, boundary_idx, lamb):
        '''
        Quantize the whole layer.

        Parameters
        -----------
        W : torch.Tensor 
            The weights for the layer.
        Q : torch.Tensor 
            The quantized weights with same shape as W.
        U : torch.Tensor 
            Quantization error matrix.
        neuron_idx: int
            The position of the neuron in the layer.
        analog_layer_input: numpy.array,
            The input for the layer of analog network.
        quantized_layer_input: numpy.array,
            The input for the layer of quantized network.
        m : int
            The batch size (num of input).
        step_size: float
            The step size of the alphabet
        boundary_idx: int
            The max idx of the alphebt to not go over
        reg: str
            The type of regularizer to be used.
        lamb: float
            The lambda for regularization.
        stochastic_quantization: bool
            Whether or not to use stochastic quantization
        '''

        for t in tqdm(range(W.shape[1])):
            U += W[:, t].unsqueeze(1) * analog_layer_input[:, t].unsqueeze(0)
            norm = torch.linalg.norm(quantized_layer_input[:, t], 2) ** 2
            if norm > 0:
                q_arg = U.matmul(quantized_layer_input[:, t]) / norm
            else: 
                q_arg = torch.zeros_like(U[:, 0])
            Q[:, t] = quantizer(step_size, q_arg, boundary_idx, lamb)
            U -= Q[:, t].unsqueeze(1) * quantized_layer_input[:, t].unsqueeze(0)


    def _quantize_layer(W, analog_layer_input, quantized_layer_input, m, 
                        step_size, boundary_idx, percentile,
                        reg, lamb, groups, stochastic_quantization, device):
        '''
        Quantize one layer in parallel.

        Parameters
        -----------
        W : torch.Tensor
            The layer weights to be quantized.
        analog_layer_input: numpy.array,
            The input for the layer of analog network.
        quantized_layer_input: numpy.array,
            The input for the layer of quantized network.
        m : int
            The batch size (num of input).
        alphabet : numpy.array
            Scalar numpy array listing the alphabet to perform quantization.
        percentile: float
            The percentile to take from each layer.
        reg: str
            The type of regularizer to be used.
        lamb: float
            The lambda for regularization.
        groups: int
            Num of grouped convolution that is used (only for Conv layers).
        stochastic_quantization: bool
            Whether or not to use stochastic quantization
        device: torch.device
            CUDA or CPU
            
        Returns
        -------
        numpy.array
            The quantized layer.
        float
            The quantize error
        float
            The relative quantize error.
        '''
        rad = torch.quantile(torch.abs(W), percentile, axis=1).mean()
        step_size = step_size * rad - lamb / boundary_idx if reg == 'L0' else step_size * rad  
        
        N, d = W.shape  # N is the number of neurons, d is the neuron dimension
        Q = torch.zeros_like(W) # quantized weights
        U = torch.zeros(N, m).to(device)   # quantization error vectors
        
        if reg == 'L1':
            quantizer = StepAlgorithm._soft_thresholding_msq
            
        elif reg == 'L0':
            quantizer = StepAlgorithm._hard_thresholding_msq
        
        else:
            if stochastic_quantization:
                quantizer = StepAlgorithm._stochastic_msq
            else:
                quantizer = StepAlgorithm._msq
                
        print(f'The number of groups: {groups}\n')
        
        if groups == 1: # no group convolutio
            StepAlgorithm._quantization(W, Q, U, analog_layer_input, quantized_layer_input, quantizer, 
                      step_size, boundary_idx, lamb)
            
            quantize_adder = U.T
            relative_adder = torch.linalg.norm(quantize_adder, axis=0) / (torch.linalg.norm(analog_layer_input @ W.T, axis=0) + 1e-5)
            quantize_error = torch.linalg.norm(quantize_adder, ord='fro')
            relative_quantize_error = quantize_error / torch.linalg.norm(analog_layer_input @ W.T, ord='fro')

        else:
            # Q shape = (out_channels, in_channels/groups*k_size[0]*k_size[1])
            W = W.view(groups, -1, W.shape[-1]) 
            Q = Q.view(groups, -1, Q.shape[-1]) #  shape (groups, out_channels/groups, in_channesl/groups*k_size[0]*k_size[1])
            U = U.view(groups, -1, U.shape[-1]) #  shape (groups, out_channels/groups, m)
            
            dims = analog_layer_input.shape # shape (B*L, in_channels*kernel_size[0]*kernel_size[1])
            analog_layer_input = analog_layer_input.view(dims[0], groups, -1)
            # shape (B*L, groups, in_channels/groups*kernel_size[0]*kernel_size[1])
            quantized_layer_input = quantized_layer_input.view(dims[0], groups, -1)

            quantize_error = 0  
            relative_quantize_error = 0

            for i in range(groups):
                StepAlgorithm._quantization(W[i], Q[i], U[i], analog_layer_input[:,i,:], quantized_layer_input[:,i,:], quantizer, 
                      step_size, boundary_idx, lamb)
                
                quantize_error += torch.linalg.norm(U[i].T, ord='fro') 
                relative_quantize_error += torch.linalg.norm(U[i].T, ord='fro') / torch.linalg.norm(analog_layer_input[:,i,:] @ W[i].T, ord='fro')
                
            quantize_error = quantize_error / groups
            relative_quantize_error = relative_quantize_error / groups
            quantize_adder = None 
            relative_adder = None
            
            Q = Q.view(-1, Q.shape[-1])
        
        return Q, quantize_error, relative_quantize_error, quantize_adder, relative_adder
                
