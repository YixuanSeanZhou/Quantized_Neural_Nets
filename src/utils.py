import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import pickle
from tqdm import tqdm

from torchvision.models.resnet import BasicBlock as tBasicBlock
from torchvision.models.resnet import Bottleneck as tBottleneck 
from torchvision.models.resnet import ResNet as tResNet
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.efficientnet import Conv2dNormActivation, SqueezeExcitation, MBConv 
from torchvision.models.mobilenetv2 import InvertedResidual

SUPPORTED_LAYER_TYPE = {nn.Linear, nn.Conv2d}
SUPPORTED_BLOCK_TYPE = {nn.Sequential, 
                        tBottleneck, tBasicBlock, tResNet,
                        BasicConv2d, Inception, InceptionAux,
                        Conv2dNormActivation, SqueezeExcitation, MBConv,
                        InvertedResidual
                        }

class InterruptException(Exception):
    pass


def parse_imagenet_val_labels(data_dir):
    """
    Generate labels of imagenet validation dataset
    More details, see 
    https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html
    """
    meta_path = os.path.join(data_dir, 'meta.mat')
    meta = sio.loadmat(meta_path, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
    idcs, wnids = list(zip(*meta))[:2]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}

    val_path = os.path.join(data_dir, 'ILSVRC2012_validation_ground_truth.txt')
    val_idcs = np.loadtxt(val_path) 
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
   
    label_path = os.path.join(data_dir, 'wnid_to_label.pickle')  
    with open(label_path, 'rb') as f:
        wnid_to_label = pickle.load(f)
    
    val_labels = [wnid_to_label[wnid] for wnid in val_wnids]
    return np.array(val_labels)


def test_accuracy(model, test_dl, device, topk=(1, )):
    """ 
    Compute top k accuracy on testing dataset
    """
    model.eval()
    maxk = max(topk)
    topk_count = np.zeros((len(topk), len(test_dl)))
    
    for j, (x_test, target) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            y_pred = model(x_test.to(device))
        topk_pred = torch.topk(y_pred, maxk, dim=1).indices
        target = target.to(device).view(-1, 1).expand_as(topk_pred)
        correct_mat = (target == topk_pred)

        for i, k in enumerate(topk):
            topk_count[i, j] = correct_mat[:, :k].reshape(-1).sum().item()

    topk_accuracy = topk_count.sum(axis=1) / len(test_dl.dataset)
    return topk_accuracy

            
def extract_layers(model, layer_list, supported_block_type=SUPPORTED_BLOCK_TYPE, supported_layer_type=SUPPORTED_LAYER_TYPE):
    '''
    Recursively obtain layers of given network
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all MLP and CNN layers
    layer_list: list
        list containing all supported layers
    '''
    for layer in model.children():
        if type(layer) in supported_block_type:
            # if sequential layer, apply recursively to layers in sequential layer
            extract_layers(layer, layer_list, supported_block_type, supported_layer_type)
        if not list(layer.children()) and type(layer) in supported_layer_type:
            # if leaf node, add it to list
            layer_list.append(layer) 


def fusion_layers_inplace(model, device):
    '''
    Let a convolutional layer fuse with its subsequent batch normalization layer  
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all CNN and BN layers
    '''
    model_layers = []
    extract_layers(model, model_layers, supported_layer_type = [nn.Conv2d, nn.BatchNorm2d])

    if len(model_layers) < 2:
        return 
    
    for i in range(len(model_layers)-1):
        curr_layer, next_layer = model_layers[i], model_layers[i+1]

        if isinstance(curr_layer, nn.Conv2d) and isinstance(next_layer, nn.BatchNorm2d):
            cnn_layer, bn_layer = curr_layer, next_layer
            # update the weight and bias of the CNN layer 
            bn_scaled_weight = bn_layer.weight.data / torch.sqrt(bn_layer.running_var + bn_layer.eps)
            bn_scaled_bias = bn_layer.bias.data - bn_layer.weight.data * bn_layer.running_mean / torch.sqrt(bn_layer.running_var + bn_layer.eps)
            cnn_layer.weight.data = cnn_layer.weight.data * bn_scaled_weight[:, None, None, None]
            # update the parameters in the BN layer 
            bn_layer.running_var = torch.ones(bn_layer.num_features, device=device)
            bn_layer.running_mean = torch.zeros(bn_layer.num_features, device=device)
            bn_layer.weight.data = torch.ones(bn_layer.num_features, device=device)
            bn_layer.eps = 0.

            if cnn_layer.bias is None:
                bn_layer.bias.data = bn_scaled_bias
            else:
                cnn_layer.bias.data = cnn_layer.bias.data * bn_scaled_weight + bn_scaled_bias 
                bn_layer.bias.data = torch.zeros(bn_layer.num_features, device=device)
            

def eval_sparsity(model):
    '''
    Compute the propotion of 0 in a network.
    
    Parameters
    ----------
    model: nn.Module
        The module to evaluate sparsity
    
    Returns
    -------
    A float capturing the proption of 0 in all the considered params.
    '''
    layers = []
    extract_layers(model, layers)
    supported_layers = [l for l in layers if type(l) in SUPPORTED_LAYER_TYPE]
    total_param = 0
    num_of_zero = 0
    
    for l in supported_layers:
        if l.weight is not None:
            total_param += l.weight.numel()
            num_of_zero += l.weight.eq(0).sum().item()
        if l.bias is not None:
            total_param += l.bias.numel()
            num_of_zero += l.bias.eq(0).sum().item()
    return np.around(num_of_zero / total_param, 4)
                
