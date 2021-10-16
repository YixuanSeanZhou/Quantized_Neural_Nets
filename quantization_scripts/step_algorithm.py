from __future__ import annotations

import numpy as np
import h5py
import multiprocessing as mp

class StepAlgorithm:
    
    def _nearest_alphabet(target_val: float, alphabet: np.array) -> float:
        '''
        Return the aproximated result to the target by the alphabet.

        Parameters
        ----------
        target_val : float
            The target value to appoximate by the alphabet.
        alphabet : np.array
            Scalar numpy array listing the alphabet to perform quantization.

        Returns
        -------
        float
            The element within the alphabet that is cloest to the target.
        '''
        
        return alphabet[np.argmin(abs(alphabet-target_val))]


    def _quantize_weight(w: float, u: float,
                         X: np.array, X_tilde: np.array,
                         alphabet: np.array) -> float:
        '''
        Quantize a particular weight parameter.

        Parameters
        -----------
        w : float
            The weight of the analog network.
        u : np.array ,
            Residual vector of the previous step.
        X : np.array
            Input to the current neuron\
            generated from the previous layer of the analog network.
        X_tilde : np.array
            Input to the current neuron\
            generated from the previous layer of the quantized network.
        alphabet : np.array
            Scalar numpy array listing the alphabet to perform quantization.

        Returns
        -------
        float
            The quantized value.
        '''

        # TODO: Is this simplification even necessary?
        if np.linalg.norm(X_tilde, 2) < 10 ** (-16):
            return StepAlgorithm._nearest_alphabet(0, alphabet)
        
        if abs(np.dot(X_tilde, u)) < 10 ** (-10):
            return StepAlgorithm._nearest_alphabet(w, alphabet)
        
        target_val = np.dot(X_tilde, u + w * X) / (np.norm(X_tilde, 2) ** 2)
        
        return StepAlgorithm._nearest_alphabet(target_val, alphabet)

    
    def _quantize_neuron(w: np.array, neuron_idx: int, input_file: str,
                         m: int, alphabet: np.array) -> np.array:
        '''
        Quantize one neuron of a layer.

        Parameters
        -----------
        w : np.array
            The single neuron.
        neuron_idx: int
            The position of the neuron in the layer.
        input_file: str
            The h5py file name of the input.
        m : int
            The batch size (num of input).
        alphabet : np.array
            Scalar numpy array listing the alphabet to perform quantization.

        Returns
        -------
        np.array
            The quantized neuron.
        '''
        
        with h5py.File(input_file, 'r') as hf:
            q = np.zeros(w.shape[0])
            u = np.zeros(m)
            for t in range(w.shape[0]):
                X_analog, X_quantize = hf['wX'][t, :], hf['qX'][t, :]
                q[t] = StepAlgorithm._quantize_weight(w[t], u, 
                                                      X_analog, X_quantize,
                                                      alphabet)
                u += w[t] * X_analog - q[t] * X_quantize

        return neuron_idx, q


    def _quantize_layer(W: np.array, input_file: str,
                        m: int, alphabet: np.array) -> np.array:
        '''
        Quantize one layer in parallel.

        Parameters
        -----------
        W : np.array
            The layer to be quantized.
        input_file: str
            The h5py file name of the input.
        m : int
            The batch size (num of input).
        alphabet : np.array
            Scalar numpy array listing the alphabet to perform quantization.

        Returns
        -------
        np.array
            The quantized layer.
        '''
        # TODO: populate file

        # Start quantize 
        pool = mp.Pool(mp.cpu_count())

        # FIXME: This defeats the purpose, partially
        # radius
        rad = alphabet * np.median(np.abs(W.flatten()))
        layer_alphabet = rad * alphabet

        Q = np.zeros(W.shape)
        results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
                                    args=(w, i, input_file, m, layer_alphabet)) 
                                    for i, w in enumerate(W.T)]
        # join
        for _ in range(Q.shape[1]):
            idx, q = results.get()
            Q[:, idx] = q

        return Q

        




