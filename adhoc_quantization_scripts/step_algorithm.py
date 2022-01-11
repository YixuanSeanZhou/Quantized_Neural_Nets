from __future__ import annotations

import numpy as np
import multiprocessing as mp

import gc

from tqdm import tqdm

class StepAlgorithm:

    def bias_correction(analog_input, quantize_input, W, Q, b, m):
        '''
        TODO: later doc
        '''
        print(m)
        gap = analog_input @ W.T - quantize_input @ Q.T
        print(gap.shape)
        # target = gap.reshape(-1)
        # A = np.vstack([np.identity(b.shape[0])] * m)
        # ret = np.linalg.lstsq(A, target)
        # A_inv = np.linalg.pinv(A)
        # b_q = A_inv @ target
        b_q = b + np.mean(gap, axis=0)
        return b_q