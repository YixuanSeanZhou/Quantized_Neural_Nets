from __future__ import annotations

import numpy as np
import multiprocessing as mp

import gc

from tqdm import tqdm

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
        # if regular alphabet

        # Z 

        # round(Z) then truncate

        # if rounded result > max, set to max
        # if round results < -max, set to -max

        # x / c
        # multiply by boolean expression

        # (abs(x) <= max) * rounded(x) + (abs(x) > max) * sign(x) * max
        # rescale at the end

        # work for odd alphabets

        # even alphabet use the floor

        # q=de*floor(u/de)+de/2;              u is x
        # q=q.*(abs(u)<=(K-1/2)*de) + sign(q).*(K-1/2).*de.*(abs(u)> (K-1/2)*de);

        # [-3, -1, 1, 3]
        # delta = 2

        # 1.7

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

        del X_analog
        del X_quantize
        gc.collect()
        return neuron_idx, q


    # def _quantize_weight_mtx(W, analog_layer_input, quantized_layer_input, m, 
    #                          alphabet, percentile, rad, surpress=False):
    #     '''
    #     Quantize one layer in parallel.
    #     Parameters
    #     -----------
    #     W : numpy.array
    #         The layer to be quantized.
    #     analog_layer_input: numpy.array,
    #         The input for the layer of analog network.
    #     quantized_layer_input: numpy.array,
    #         The input for the layer of quantized network.
    #     m : int
    #         The batch size (num of input).
    #     alphabet : numpy.array
    #         Scalar numpy array listing the alphabet to perform quantization.
    #     percentile: float
    #         The percentile to take from each layer.
    #     rad: float
    #         The radius of the alphabet for this layer.
    #     surpress: bool
    #         Whether or not to surpress the tqdm.
    #     Returns
    #     -------
    #     numpy.array
    #         The quantized layer.
    #     '''
    #     # Start quantize 
    #     pool = mp.Pool(mp.cpu_count() - 1)

    #     # FIXME: This defeats the purpose, partially
    #     # May move the layer_alphabet to quantize_neural_net.py
    #     # rad = np.median(np.abs(W))  # radius
        
    #     layer_alphabet = alphabet * rad 

    #     Q = np.zeros_like(W)
    #     results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
    #                                 args=(w, i, analog_layer_input, 
    #                                       quantized_layer_input, m,
    #                                       layer_alphabet)) 
    #                                 for i, w in enumerate(W)]
    #     # join
    #     if surpress:
    #         for i in range(Q.shape[0]):
    #             idx, q = results[i].get()
    #             Q[idx, :] = q
    #     else:
    #         for i in tqdm(range(Q.shape[0])):
    #             idx, q = results[i].get()
    #             Q[idx, :] = q

    #     pool.close()
        
    #     del pool

    #     gc.collect()
    #     return Q
    

    def _quantize_layer(W, b, analog_layer_input, quantized_layer_input, m, 
                        alphabet, percentile, groups=1):
        '''
        Quantize one layer in parallel.
        Parameters
        -----------
        W : numpy.array
            The layer to be quantized.
        b : numpy.array
            The layer's bias to be corrected
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
        groups: int
            Num of grouped convolution that is used (only for Conv layers).
        Returns
        -------
        numpy.array
            The quantized layer.
        float
            The quantize error
        float
            The relative quantize error.
        '''
        pool = mp.Pool(mp.cpu_count() - 1)
        
        b_q = b
        
        rad = np.quantile(np.abs(W), percentile, axis=1).mean()
        layer_alphabet = alphabet * rad 

        Q = np.zeros_like(W)
        
        if groups == 1: # no group convolution
            results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
                                        args=(w, idx, analog_layer_input, 
                                            quantized_layer_input, m,
                                            layer_alphabet)) 
                                        for idx, w in enumerate(W)]
            # join
            for i in tqdm(range(Q.shape[0])):
                idx, q = results[i].get()
                Q[idx, :] = q

            quantize_error = np.linalg.norm(analog_layer_input @ W.T  
                            - quantized_layer_input @ Q.T, ord='fro')
            relative_quantize_error = quantize_error / np.linalg.norm(analog_layer_input @ W.T, ord='fro')
            
            # bias correction
            
            if b is not None:
                b_q = StepAlgorithm.bias_correction(analog_layer_input, 
                                                    quantized_layer_input, 
                                                    W, Q, b, m)
                
                uncorrected_layer_error = np.linalg.norm(analog_layer_input @ W.T + b 
                                - quantized_layer_input @ Q.T + b, ord='fro')
                corrected_layer_error = np.linalg.norm(analog_layer_input @ W.T + b 
                                - quantized_layer_input @ Q.T + b_q, ord='fro')
                
                print(f'Uncorrected Quantize Error: {uncorrected_layer_error}')
                print(f'Corrected Quantize Error: {corrected_layer_error}')
            

        else:
            # Q = np.zeros_like(W) # shape (out_channels, in_channels/groups*k_size[0]*k_size[1])
            W = W.reshape(groups, -1, W.shape[-1]) 
            Q = Q.reshape(groups, -1, Q.shape[-1]) 
            #  shape (groups, out_channels/groups, in_channesl/groups*k_size[0]*k_size[1])
            dims = analog_layer_input.shape # shape (B*L, in_channels*kernel_size[0]*kernel_size[1])
            analog_layer_input = analog_layer_input.reshape(dims[0], groups, -1)
            # shape (B*L, groups, in_channels/groups*kernel_size[0]*kernel_size[1])
            quantized_layer_input = quantized_layer_input.reshape(dims[0], groups, -1)

            analog_output_norms = []
            quantize_error = 0

            results = []

            for i in range(groups):
                group_results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
                                                  args=(w, idx, 
                                                        analog_layer_input[:,i,:], 
                                                        quantized_layer_input[:,i,:], 
                                                        m, layer_alphabet)) 
                                                    for idx, w in enumerate(W[i])]
                results.append(group_results)

            for i in tqdm(range(groups)):
                for j in range(W[i].shape[0]):
                    # Linear way
                    # idx, q = StepAlgorithm._quantize_neuron(w, idx, 
                    #                     analog_layer_input, 
                    #                     quantized_layer_input, m,
                    #                     layer_alphabet)
                    idx, q = results[i][j].get()
                    Q[i, idx] = q
                analog_output = analog_layer_input[:,i,:] @ W[i].T
                quantize_output = quantized_layer_input[:,i,:] @ Q[i].T
                quantize_error += np.linalg.norm(analog_output - quantize_output, ord='fro') ** 2
                analog_output_norms.append(np.linalg.norm(analog_output, ord='fro')**2)

            # TODO: refactor this into another layer of multi-processing if needed
            # for i in tqdm(range(groups)):
            #     # note that m = B*L = analog_layer_input[:,i,:].shape[0]
            #     Q[i] = StepAlgorithm._quantize_weight_mtx(
            #         W[i], analog_layer_input[:,i,:], quantized_layer_input[:,i,:], m,
            #         alphabet, percentile, rad,
            #         surpress=True
            #     )
            #     analog_output = analog_layer_input[:,i,:] @ W[i].T
            #     quantize_output = quantized_layer_input[:,i,:] @ Q[i].T
            #     quantize_error += np.linalg.norm(analog_output - quantize_output, ord='fro') ** 2
            #     analog_output_norms.append(np.linalg.norm(analog_output, ord='fro')**2)
            
            relative_quantize_error = np.sqrt(quantize_error / np.array(analog_output_norms).sum())
            quantize_error = np.sqrt(quantize_error)
            
            
        pool.close()    
        
        del pool

        gc.collect()
        return Q, b_q, quantize_error, relative_quantize_error


    def bias_correction(analog_input, quantize_input, W, Q, b, m):
        '''
        TODO: later doc
        '''
        print(m)
        gap = analog_input @ W.T - quantize_input @ Q.T
        print(gap.shape)
        target = gap.reshape(-1)
        A = np.vstack([np.identity(b.shape[0])] * m)
        # ret = np.linalg.lstsq(A, target)
        # A_inv = np.linalg.pinv(A)
        # b_q = A_inv @ target
        b_q = b + np.mean(gap, axis=0)
        return b_q
        
        
        

            # old code for else
            # Ws = np.split(W, groups)
            
            # print(f'There are in total {groups} group')
            # input_len = W.shape[-1]

            # Qs = []
            # analog_outputs = []
            # quantize_outputs = []
            
            # if W.shape[0] != groups: # general case for group conv
            #                          # quantize group-wise

            #     for i in range(groups):
            #         print(f'Quantize group {i}')
            #         analog_group_input = analog_layer_input[:, i * input_len: (i+1) * input_len]
            #         quantized_group_input = quantized_layer_input[:, i * input_len: (i+1) * input_len]
                    
            #         Q_group = StepAlgorithm._quantize_weight_mtx(
            #             Ws[i], 
            #             analog_group_input,
            #             quantized_group_input,
            #             m, alphabet, percentile
            #         )
            #         Qs.append(Q_group)
            #         analog_outputs.append(analog_group_input @ Ws[i].T)
            #         quantize_outputs.append(quantized_group_input @ Q_group.T)
            
            # else: # special case, in which each input channel is convolved with its own filter

            #     rad = np.quantile(np.abs(W), percentile, axis=1).mean()
            #     layer_alphabet = alphabet * rad * (groups // 16 + 1)

            #     W_group_shape = Ws[0].shape

            #     pool = mp.Pool(mp.cpu_count() - 1)

            #     results = [pool.apply_async(StepAlgorithm._quantize_neuron, 
            #                                 args=(w.reshape(-1), i, analog_layer_input[:, i * input_len: (i+1) * input_len], 
            #                                         quantized_layer_input[:, i * input_len: (i+1) * input_len], 
            #                                         m, layer_alphabet)) 
            #                                 for i, w in enumerate(Ws)]
            #     # join
            #     for i in tqdm(range(len(results))):
            #         idx, Q_group = results[i].get()
                    
            #         Q_group = Q_group.reshape(W_group_shape)
                    
            #         Qs.append(Q_group)
                    
            #         analog_group_input = analog_layer_input[:, i * input_len: (i+1) * input_len]
            #         quantized_group_input = quantized_layer_input[:, i * input_len: (i+1) * input_len]

            #         analog_outputs.append(analog_group_input @ Ws[i].T)
            #         quantize_outputs.append(quantized_group_input @ Q_group.T)

            #     pool.close()
                
            #     del pool

            #     gc.collect()

            # Q = np.vstack(Qs)

            # analog_output = np.vstack(analog_outputs)
            # quantize_output = np.vstack(quantize_outputs)

            # quantize_error = np.linalg.norm(analog_output - quantize_output, ord='fro')
            # relative_quantize_error = quantize_error / np.linalg.norm(analog_output, ord='fro')

       

        



        # SOME MSQ stuffs if we want
        # Q_temp = np.zeros_like(W)

        # for i in range(len(Q_temp)):
        #     for j in range(len(Q_temp[i])):
        #         if W[i][j] > 1 * rad / 4:
        #             Q_temp[i][j] = 1 * rad / 2
        #         elif W[i][j] < - 1 * rad / 4:
        #             Q_temp[i][j] = -1 * rad / 2
        #         else:
        #             Q_temp[i][j] = 0
        
        # msq_quantize_error = np.linalg.norm(analog_layer_input @ W.T  
        #                 - quantized_layer_input @ Q_temp.T, ord='fro')
        # print(f'msq: {msq_quantize_error}')
