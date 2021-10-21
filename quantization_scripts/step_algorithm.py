from __future__ import annotations

import numpy
import h5py
import multiprocessing as mp

ANALOG_INPUT = 'ANALOG_INPUT'
QUANTIZE_INPUT = 'QUANTIZE_INPUT'

class StepAlgorithm:
    
    def _nearest_alphabet(target_val: float, alphabet: numpy.array) -> float:
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
        
        return alphabet[numpy.argmin(abs(alphabet-target_val))]


    def _quantize_weight(w: float, u: float,
                         X: numpy.array, X_tilde: numpy.array,
                         alphabet: numpy.array) -> float:
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

        # TODO: Is this simplification even necessary?
        if numpy.linalg.norm(X_tilde, 2) < 10 ** (-16):
            return StepAlgorithm._nearest_alphabet(0, alphabet)
        
        if abs(numpy.dot(X_tilde, u)) < 10 ** (-10):
            return StepAlgorithm._nearest_alphabet(w, alphabet)
        
        target_val = numpy.dot(X_tilde, u + w * X) / (numpy.linalg.norm(X_tilde, 2) ** 2)
        
        return StepAlgorithm._nearest_alphabet(target_val, alphabet)

    
    def _quantize_neuron(w: numpy.array, neuron_idx: int, 
                         analog_layer_input: numpy.array,
                         quantized_layer_input: numpy.array,
                         m: int, alphabet: numpy.array) -> numpy.array:
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
        q = numpy.zeros(w.shape[0])
        u = numpy.zeros(m)
        for t in range(w.shape[0]):
            X_analog = analog_layer_input[:, t]
            X_quantize = quantized_layer_input[:, t]
            q[t] = StepAlgorithm._quantize_weight(w[t], u, 
                                                    X_analog, X_quantize,
                                                    alphabet)
            u += w[t] * X_analog - q[t] * X_quantize

        print(numpy.linalg.norm(u))

        return neuron_idx, q


    def _quantize_layer(W: numpy.array,
                        analog_layer_input: numpy.array,
                        quantized_layer_input: numpy.array,
                        m: int, alphabet: numpy.array,
                        ) -> numpy.array:
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
        # radius
        rad = alphabet * numpy.median(numpy.abs(W.flatten()))
        layer_alphabet = rad * alphabet

        Q = numpy.zeros(W.shape)
        results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
                                    args=(w, i, analog_layer_input, 
                                          quantized_layer_input, m,
                                          layer_alphabet)) 
                                    for i, w in enumerate(W.T)]
        # join
        for i in range(Q.shape[1]):
            idx, q = results[i].get()
            Q[:, idx] = q

        # for i, w in enumerate(W):
        #     idx, q = StepAlgorithm._quantize_neuron(w, i, 
        #                                         analog_layer_input, 
        #                                         quantized_layer_input,
        #                                         m , layer_alphabet)
        #     Q[idx, :] = q

        pool.close()

        return Q


        




