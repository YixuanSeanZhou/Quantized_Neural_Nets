from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils import data
import torchvision
from torchvision import transforms
from d2l import torch as d2l
import multiprocessing as mp
import numpy

from data_loaders import load_data_mnist


class MLP(nn.Module):
    '''
    The most navie MLP network with input 28*28 -> 256 -> 10 -> softmax
    '''
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 256)
        self.layer_2 = nn.Linear(256, 10)

    def forward(self, X: torch.tensor):
        X = X.view(-1, 28 * 28)
        X = self.layer_1(X)
        X = functional.relu(X)
        X = self.layer_2(X)
        return functional.log_softmax(X, dim=0)


def train_mlp(model: nn.Module, lr: float, 
              train_loader: data.DataLoader, 
              torch_loss_function: type, 
              torch_optimizer: type,
              ) -> nn.Module:
    
    loss_function = torch_loss_function()
    optimizer = torch_optimizer(model.parameters(), lr=lr)

    model.train()
    for epoch in range(0, 2):
        print(f'Starting epoch {epoch +1}')
        current_loss = 0.0
        for i, (features, labels) in enumerate(train_loader, 0):
            
            optimizer.zero_grad()
            
            results = model(features)
            
            # Compute loss
            loss = loss_function(results, labels)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0

def test_mlp(model: nn.Module, 
             test_loader: data.DataLoader,
             torch_loss_function: function):
    loss_function = torch_loss_function()
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()

    for i, (features, labels) in enumerate(test_loader, 0):

        output = model(features)

        loss = loss_function(output, labels)

        test_loss += loss.item()

        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = numpy.squeeze(pred.eq(labels.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    # calculate and print avg test loss
    test_loss = test_loss/i
    print('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                numpy.sum(class_correct[i]), numpy.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * numpy.sum(class_correct) / numpy.sum(class_total),
        numpy.sum(class_correct), numpy.sum(class_total)))


if __name__ == '__main__':
    train_loader, test_loader = load_data_mnist(50)
    mlp = MLP()
    loss_function = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    train_mlp(mlp, 0.01, train_loader, loss_function, optimizer)
    test_mlp(mlp, test_loader, loss_function)
    torch.save(mlp, '../models/mlp.pt')
 


