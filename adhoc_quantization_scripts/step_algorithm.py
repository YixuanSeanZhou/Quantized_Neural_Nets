from __future__ import annotations

import numpy as np
import multiprocessing as mp

import gc

from tqdm import tqdm

class StepAlgorithm:

    def bias_correction(analog_input, quantize_input, W, Q, b, m):
        '''
        Bias correct the layer. 
        
        Parameters:
        -----------
        analog_input: numpy.array
            The input to the analog layer
        quantize_input: numpy.array
            The input to the quantize layer
        W: numpy.array
            The weight matrix of the analog layer
        Q: numpy.array
            The weight matrix of the quantzied layer
        b: numpy.array
            The bias term of the analog layer
        m: int
            Batch size
        Returns
        -------
        b_q
            The corrected bias of the quantized layer using average.
        '''
        gap = analog_input @ W.T - quantize_input @ Q.T
        b_q = b + np.mean(gap, axis=0)
        return b_q
        