from __future__ import annotations

import numpy as np


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
        u : array ,
            Residual vector of the previous step.
        X : array
            Input to the current neuron\
            generated from the previous layer of the analog network.
        X_tilde : array
            Input to the current neuron\
            generated from the previous layer of the quantized network.
        alphabet : array
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




