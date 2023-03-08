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
from utils import parse_imagenet_val_labels

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


def data_loader(ds_name, batch_size, num_workers): 
    """
    Prepare data loaders
    """
    if ds_name == 'ILSVRC2012':
        data_dir = '../data/ILSVRC2012'  # customize the data path before run the code 

        if not os.path.isdir(data_dir):
            raise Exception('Please download Imagenet2012 dataset!')

        # see https://pytorch.org/vision/stable/models.html for setting transform
        transform = transforms.Compose([
                            transforms.Resize(256), 
                            transforms.CenterCrop(224),  
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                            ])
        
        train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_train'),
                                                    transform=transform)
        
        if not os.path.isfile('../data/ILSVRC2012/wnid_to_label.pickle'):
            with open('../data/ILSVRC2012/wnid_to_label.pickle', 'wb') as f:
                pickle.dump(train_ds.class_to_idx, f)         

        test_ds = Imagenet(data_dir, transform) 
        # test_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_val'),
        #                                             transform=transform)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=g)
        test_dl = DataLoader(test_ds, min(batch_size, 1024), shuffle=False,
                                num_workers=num_workers) 

    elif ds_name == 'CIFAR10':
        data_dir = '../data'

        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, 
            transform=transform_train)
        
        test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, 
            transform=transform_test)
        
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=min(batch_size, 1024),
                             num_workers=num_workers)

    else:
        raise Exception('Unkown dataset!')

    return train_dl, test_dl 
