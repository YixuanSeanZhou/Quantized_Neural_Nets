import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import multiprocessing as mp

def get_dataloader_workers():
    return mp.cpu_count() - 1


def load_data_fashion_mnist(batch_size, resize=None, train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST("../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST("../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    train_size = int(len(mnist_train) * train_ratio)
    test_size =  len(mnist_train) - train_size
    # Split dataset and the generator is used for reproducible results:    
    train_data, val_data = random_split(mnist_train, [train_size, test_size], 
                                generator=torch.Generator().manual_seed(42))   
    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                        num_workers=num_workers)
    val_loader =  DataLoader(val_data, batch_size, shuffle=False,
                        num_workers=num_workers)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False,
                            num_workers=num_workers)             
    return train_loader, val_loader, test_loader


def load_data_mnist(batch_size, resize=None, train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST("../data",
                                             train=True,
                                             transform=trans,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST("../data",
                                            train=False,
                                            transform=trans,
                                            download=True)
    train_size = int(len(mnist_train) * train_ratio)
    test_size =  len(mnist_train) - train_size
    # Split dataset and the generator is used for reproducible results:    
    train_data, val_data = random_split(mnist_train, [train_size, test_size], 
                                generator=torch.Generator().manual_seed(42))   
    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                        num_workers=num_workers)
    val_loader =  DataLoader(val_data, batch_size, shuffle=False,
                        num_workers=num_workers)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False,
                            num_workers=num_workers)             
    return train_loader, val_loader, test_loader
