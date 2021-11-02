import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import multiprocessing as mp



def get_dataloader_workers():
    return mp.cpu_count() - 1

def load_data_fashion_mnist(batch_size, transform=[transforms.ToTensor()], train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the Fashion-MNIST dataset and then load it into memory."""
    transform = transforms.Compose(transform)
    mnist_train = torchvision.datasets.FashionMNIST("../data",
                                                    train=True,
                                                    transform=transform,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST("../data",
                                                   train=False,
                                                   transform=transform,
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


def load_data_mnist(batch_size, transform=[transforms.ToTensor()], train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the MNIST dataset and then load it into memory."""
    transform = transforms.Compose(transform)
    mnist_train = torchvision.datasets.MNIST("../data",
                                             train=True,
                                             transform=transform,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST("../data",
                                            train=False,
                                            transform=transform,
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


def load_data_kmnist(batch_size, transform=[transforms.ToTensor()], train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the K-MNIST dataset and then load it into memory."""
    transform = transforms.Compose(transform)
    mnist_train = torchvision.datasets.KMNIST("../data",
                                                    train=True,
                                                    transform=transform,
                                                    download=True)
    mnist_test = torchvision.datasets.KMNIST("../data",
                                                   train=False,
                                                   transform=transform,
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


def load_data_cifar10(batch_size, transform=[transforms.ToTensor()], train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the CIFAR10 dataset and then load it into memory."""
    transform = transforms.Compose(transform)
    mnist_train = torchvision.datasets.CIFAR10("../data",
                                                    train=True,
                                                    transform=transform,
                                                    download=True)
    mnist_test = torchvision.datasets.CIFAR10("../data",
                                                   train=False,
                                                   transform=transform,
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


def load_data_cifar100(batch_size, transform=[transforms.ToTensor()], train_ratio=0.8, num_workers=get_dataloader_workers()): 
    """Download the CIFAR100 dataset and then load it into memory."""
    transform = transforms.Compose(transform)
    mnist_train = torchvision.datasets.CIFAR100("../data",
                                                    train=True,
                                                    transform=transform,
                                                    download=True)
    mnist_test = torchvision.datasets.CIFAR100("../data",
                                                   train=False,
                                                   transform=transform,
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

