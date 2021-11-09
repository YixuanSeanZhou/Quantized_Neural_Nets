import torch
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os
import pickle

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


def test_accuracy(model, test_dl, topk=(1, )):
    model.eval()
    maxk = max(topk)
    topk_count = np.zeros((len(topk), len(test_dl)))
    
    for j, (x_test, target) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            y_pred = model(x_test)
        topk_pred = torch.topk(y_pred, maxk, dim=1).indices
        target = target.view(-1, 1).expand_as(topk_pred)
        correct_mat = (target == topk_pred)

        for i, k in enumerate(topk):
            topk_count[i, j] = correct_mat[:, :k].reshape(-1).sum().item()

    topk_accuracy = topk_count.sum(axis=1) / len(test_dl.dataset)
    return topk_accuracy