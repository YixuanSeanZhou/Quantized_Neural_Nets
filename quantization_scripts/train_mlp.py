from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from data_loaders import load_data_mnist, load_data_fashion_mnist

# If we need to train complicated models, then we enable GPUs.
# Will do this later.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'{device} is available.')


class MLP(nn.Module):
    '''
    The most navie MLP network with input 28*28 -> 256 -> 10 -> softmax
    '''
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # Define layers of MLP
        self.layer1 = nn.Linear(input_dim, hidden_dim[0], bias=False)
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1], bias=False)
        self.layer3 = nn.Linear(hidden_dim[1], out_dim, bias=False)

    def forward(self, X):
        X = X.view(-1, self.input_dim)
        X = self.layer1(X)
        X = F.relu(X)
        X = self.layer2(X)
        X = F.relu(X)
        X = self.layer3(X)
        return F.log_softmax(X, dim=1)

def train_mlp(train_loader, val_loader, model, loss_fn, optimizer, n_epochs):
    """
    Add annotations as docstrings. Will do this later.
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(1, n_epochs+1):
        model.train()
        batch_losses = []
        for x_batch, y_batch in train_loader:
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())
        training_loss = np.mean(batch_losses)
        train_losses.append(training_loss)

        model.eval()  # Start validation
        batch_val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_output = model(x_val)
                val_loss = loss_fn(val_output, y_val).item()
                batch_val_losses.append(val_loss)
            validation_loss = np.mean(batch_val_losses)
            val_losses.append(validation_loss)
        
        if (epoch <= 10) | (epoch % 10 == 0):
            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t \
                Validation loss: {validation_loss:.4f}"
            )
    return train_losses, val_losses


def test_mlp(model: nn.Module, 
             test_loader: data.DataLoader,
             torch_loss_function: function):
    loss_function = torch_loss_function()
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()

    for i, (features, labels) in enumerate(test_loader):

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
    batch_size = 16
    input_dim = 28 * 28
    hidden_dim = [512, 256]
    out_dim = 10
    n_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-6 

    train_loader, val_loader, test_loader = load_data_mnist(batch_size, train_ratio=0.8)
    model = MLP(input_dim, hidden_dim, out_dim)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, val_losses = train_mlp(train_loader, val_loader, model, loss_fn, optimizer, n_epochs)
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.show()

    # test_mlp(mlp, test_loader, loss_function)
    torch.save(model.state_dict(), '../models/mlp.pt')