from __future__ import annotations

import numpy as np
import multiprocessing as mp


class StepAlgorithm:
    
    def _nearest_alphabet(target_val, alphabet):
        '''
        Return the aproximated result to the target by the alphabet.
        Parameters
        ----------
        target_val : float
            The target value to appoximate by the alphabet.
        alphabet : numpy.array
            Scalar numpy array listing the alphabet to perform quantization.
        Returns
        -------
        float
            The element within the alphabet that is cloest to the target.
        '''
        return alphabet[np.argmin(np.abs(alphabet-target_val))]


    def _quantize_weight(w, u, X_analog, X_quantize, alphabet):
        '''
        Quantize a particular weight parameter.
        Parameters
        -----------
        w : float
            The weight of the analog network.
        u : numpy.array ,
            Residual vector of the previous step.
        X : numpy.array
            Input to the current neuron\
            generated from the previous layer of the analog network.
        X_tilde : numpy.array
            Input to the current neuron\
            generated from the previous layer of the quantized network.
        alphabet : numpy.array
            Scalar numpy array listing the alphabet to perform quantization.
        Returns
        -------
        float
            The quantized value.
        '''

        if np.linalg.norm(X_quantize, 2) < 10 ** (-16):
            return StepAlgorithm._nearest_alphabet(0, alphabet)
        
        if abs(np.dot(X_quantize, u)) < 10 ** (-10):
            return StepAlgorithm._nearest_alphabet(w, alphabet)

        target_val = np.dot(X_quantize, u + w * X_analog) / (np.linalg.norm(X_quantize, 2) ** 2)
        return StepAlgorithm._nearest_alphabet(target_val, alphabet)

    
    def _quantize_neuron(w, neuron_idx, analog_layer_input, quantized_layer_input,
                         m, alphabet):
        '''
        Quantize one neuron of a layer.
        Parameters
        -----------
        w : numpy.array
            The single neuron.
        neuron_idx: int
            The position of the neuron in the layer.
        analog_layer_input: numpy.array,
            The input for the layer of analog network.
        quantized_layer_input: numpy.array,
            The input for the layer of quantized network.
        m : int
            The batch size (num of input).
        alphabet : numpy.array
            Scalar numpy array listing the alphabet to perform quantization.
        Returns
        -------
        numpy.array
            The quantized neuron.
        '''
        q = np.zeros(len(w))
        u = np.zeros(m)
        for t in range(len(w)):
            X_analog = analog_layer_input[:, t]
            X_quantize = quantized_layer_input[:, t]
            q[t] = StepAlgorithm._quantize_weight(w[t], u, 
                                                    X_analog, X_quantize,
                                                    alphabet)
            u += w[t] * X_analog - q[t] * X_quantize

        return neuron_idx, q


    def _quantize_layer(W, analog_layer_input, quantized_layer_input, m, alphabet):
        '''
        Quantize one layer in parallel.
        Parameters
        -----------
        W : numpy.array
            The layer to be quantized.
        analog_layer_input: numpy.array,
            The input for the layer of analog network.
        quantized_layer_input: numpy.array,
            The input for the layer of quantized network.
        m : int
            The batch size (num of input).
        alphabet : numpy.array
            Scalar numpy array listing the alphabet to perform quantization.
        Returns
        -------
        numpy.array
            The quantized layer.
        '''
        # Start quantize 
        pool = mp.Pool(mp.cpu_count())

        # FIXME: This defeats the purpose, partially
        # May move the layer_alphabet to quantize_neural_net.py
        # rad = np.median(np.abs(W))  # radius
        rad = np.abs(W).max()
        layer_alphabet = alphabet * rad
        # layer_alphabet = W.shape[1] *1e-2 * len(alphabet) * alphabet

        Q = np.zeros_like(W)
        results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
                                    args=(w, i, analog_layer_input, 
                                          quantized_layer_input, m,
                                          layer_alphabet)) 
                                    for i, w in enumerate(W)]
        # join
        for i in range(Q.shape[0]):
            idx, q = results[i].get()
            Q[idx, :] = q

        # for i, w in enumerate(W):
        #     idx, q = StepAlgorithm._quantize_neuron(w, i, 
        #                                         analog_layer_input, 
        #                                         quantized_layer_input,
        #                                         m , layer_alphabet)
        #     Q[idx, :] = q

        pool.close()
        quantize_error = np.linalg.norm(analog_layer_input @ W.T  
                            - quantized_layer_input @ Q.T, ord='fro')
                            
        return Q, quantize_error
