from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_loaders import load_data_mnist, load_data_fashion_mnist, load_data_kmnist

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    '''
    Basic CNN to test quantization.
    '''
    def __init__(self):
        super().__init__()

        # Define layers of MLP. Using nn.Sequential is also OK.
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # X = X.view(-1, self.input_dim)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



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
            output = model(x_batch.to(device))
            loss = loss_fn(output, y_batch.to(device))
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
                output = model(x_val.to(device))
                val_loss = loss_fn(output, y_val.to(device)).item()
                batch_val_losses.append(val_loss)
            validation_loss = np.mean(batch_val_losses)
            val_losses.append(validation_loss)
        
        if (epoch <= 20) | (epoch % 10 == 0):
            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t \
                Validation loss: {validation_loss:.4f}"
            )
    return train_losses, val_losses


def test_mlp(test_loader, model):
    """
    Add annotations later.
    """
    predictions = []
    labels = []
    model.eval()

    with torch.no_grad():
        for x_test, y_test in test_loader:
            _, pred = model(x_test.to(device)).max(dim=1)
            predictions.append(pred.cpu().numpy())
            labels.append(y_test.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return predictions, labels


if __name__ == '__main__':
    print(f'{device} is available.')
    batch_size = 16
    input_dim = 28 * 28
    hidden_dim = [512, 256, 128]
    out_dim = 10
    n_epochs = 5
    learning_rate = 5 * 1e-4
    weight_decay = 1e-6 
    num_workers = 4  # num_workers is around 4 * num_of_GPUs
    train_loader, val_loader, test_loader = load_data_kmnist(batch_size, train_ratio=0.8, 
                                                num_workers=num_workers)
    model = CNN().to(device)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, val_losses = train_mlp(train_loader, val_loader, model, loss_fn, optimizer, n_epochs)
    # Calculate testing accuracy:
    predictions, labels = test_mlp(test_loader, model)
    test_accuracy = np.sum(predictions == labels) / len(labels)
    print(f'The testing accuracy is: {test_accuracy}.')
    torch.save(model, '../models/conv2d_kmlp.pt')