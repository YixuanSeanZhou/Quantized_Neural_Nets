import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import multiprocessing as mp

def load_data():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False,
                                                   transform=trans,
                                                   download=True)
    mnist_train = torchvision.datasets.MNIST(root="../data", train=True,
                                             transform=trans,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data", train=False,
                                            transform=trans,
                                            download=True)

    print(len(mnist_test))

def get_dataloader_workers():
    # reserve one cpu and use the rest to load the data.
    return mp.cpu_count() - 1


def load_data_fashion_mnist(batch_size, resize=None): 
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_mnist(batch_size, resize=None): 
    """Download the MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root="../data",
                                             train=True,
                                             transform=trans,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data",
                                            train=False,
                                            transform=trans,
                                            download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


if __name__ == '__main__':
    load_data()
    import os
    print(os.path.abspath('.'))
