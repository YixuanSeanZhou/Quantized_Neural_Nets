import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import os
import glob
import re
import pickle
from helper_tools import parse_imagenet_val_labels

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
# use above function and g to preserve reproducibility.


class Imagenet(Dataset):
    """
    Validation dataset of Imagenet
    """
    def __init__(self, data_dir, transform):
        # we can maybe pput this into diff files.
        self.Y = torch.from_numpy(parse_imagenet_val_labels(data_dir)).long()
        self.X_path = sorted(glob.glob(os.path.join(data_dir, 'ILSVRC2012_img_val/*.JPEG')), 
            key=lambda x: re.search('%s(.*)%s' % ('ILSVRC2012_img_val/', '.JPEG'), x).group(1))
        self.transform = transform

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, idx):
        img = Image.open(self.X_path[idx]).convert('RGB')
        y = self.Y[idx] 
        if self.transform:
            x = self.transform(img)
        return x, y


def data_loader(ds_name, batch_size, transform, num_workers): 
    """
    Prepare data loaders
    """
    if ds_name == 'ILSVRC2012':
        data_dir = '../data/ILSVRC2012'

        if not os.path.isdir(data_dir):
            raise Exception('Please download Imagenet2012 dataset!')
        
        train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_train'),
                                                    transform=transform)
        
        if not os.path.isfile('../data/ILSVRC2012/wnid_to_label.pickle'):
            with open('../data/ILSVRC2012/wnid_to_label.pickle', 'wb') as f:
                pickle.dump(train_ds.class_to_idx, f)         

        test_ds = Imagenet(data_dir, transform) 
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=g)
        test_dl = DataLoader(test_ds, batch_size, shuffle=False,
                                num_workers=num_workers) 

    else:
        train_ds = getattr(torchvision.datasets, ds_name)("../data",
                                                        train=True,
                                                        transform=transform,
                                                        download=True)
        test_ds = getattr(torchvision.datasets, ds_name)("../data",
                                                    train=False,
                                                    transform=transform,
                                                    download=True)
                                    
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=g)
        test_dl = DataLoader(test_ds, batch_size, shuffle=False,
                                num_workers=num_workers)             
    return train_dl, test_dl 
