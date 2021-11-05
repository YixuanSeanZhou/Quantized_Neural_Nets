import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import random
import multiprocessing as mp
import json
import pickle

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
# use above function and g to preserve reproducibility.

def get_dataloader_workers():
    return mp.cpu_count() - 1

def data_loader(ds_name, batch_size, transform, train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download ds_name and then load it into memory."""
    if ds_name == 'MiniImagenet':
        return data_loader_miniimagenet(batch_size=batch_size, transform=transform)

    mnist_train = getattr(torchvision.datasets, ds_name)("../data",
                                                    train=True,
                                                    transform=transform,
                                                    download=True)
    mnist_test = getattr(torchvision.datasets, ds_name)("../data",
                                                   train=False,
                                                   transform=transform,
                                                   download=True)
    train_size = int(len(mnist_train) * train_ratio)
    test_size =  len(mnist_train) - train_size
    # Split dataset and the generator is used for reproducible results:    
    train_data, val_data = random_split(mnist_train, [train_size, test_size], generator=g)
                                 
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)
    val_loader =  DataLoader(val_data, batch_size, shuffle=False,
                        num_workers=num_workers)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False,
                            num_workers=num_workers)             
    return train_loader, val_loader, test_loader


class MiniImagenet(Dataset):
    def __init__(self, data, label_dict, transform=None):
        self.X = data['image_data']
        self.Y = torch.Tensor(self.label_prep(data, label_dict)).long()
        self.transform = transform
    
    def label_prep(self, data, label_dict):
        label_list = []
        for i in range(len(data['image_data'])):
            label_str = next(k for k, v in data['class_dict'].items() if i in v)
            label_list.append(label_dict[label_str])
        return label_list

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def data_loader_miniimagenet(batch_size, transform, num_workers=get_dataloader_workers()):
    with open('../data/miniimagenet/imagenet_class_index.json') as f:
            d = json.load(f) 
            label_dict = {v[0]: int(k) for k, v in d.items()}

    train_f = open('../data/miniimagenet/mini-imagenet-cache-train.pkl', 'rb')
    train_data = pickle.load(train_f)
    train_f.close()
    train_ds = MiniImagenet(train_data, label_dict, transform)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)

    val_f = open("../data/miniimagenet/mini-imagenet-cache-val.pkl", "rb")
    val_data = pickle.load(val_f)
    val_f.close()
    val_ds = MiniImagenet(val_data, label_dict, transform)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    test_f = open("../data/miniimagenet/mini-imagenet-cache-test.pkl", "rb")
    test_data = pickle.load(test_f)
    test_f.close()
    test_ds = MiniImagenet(test_data, label_dict, transform)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl
    


# MNIST
# CIFAR10
# Mini Image Net
# IMAGE NET

# AlexNet 
# VGG
# ResNet
# GoogleLeNet
