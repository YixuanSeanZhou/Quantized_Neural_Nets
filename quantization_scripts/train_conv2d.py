from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_loaders import load_data_mnist, load_data_fashion_mnist, load_data_kmnist

torch.multiprocessing.set_sharing_strategy('file_system')  # used for training on Linux based system

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    """
        Define and allocate layers for LeNet.
        Args:
            num_classes (int): number of classes to predict with this model
        """
    def __init__(self, num_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(1, 6, kernel_size=5),  # (b, 6, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # (b, 6, 12, 12)
            nn.Conv2d(6, 16, kernel_size=5),  # (b, 16, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # (b, 16, 4, 4)   
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 128),   # (b, 128)   
            nn.ReLU(),
            nn.Linear(128, 64),   # (b, 64)   
            nn.ReLU(),
            nn.Linear(64, num_classes),  # (b, 10)   
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax(logits, dim=1)
        return probs


def train(train_loader, val_loader, model, loss_fn, optimizer, n_epochs):
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


def test(test_loader, model):
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
    # input_dim = 28 * 28
    # hidden_dim = [512, 256, 128]
    num_classes = 10
    n_epochs = 15
    learning_rate = 1e-3
    weight_decay = 1e-6 
    num_workers = 4  # num_workers is around 4 * num_of_GPUs
    train_loader, val_loader, test_loader = load_data_mnist(batch_size, train_ratio=0.8, 
                                                num_workers=num_workers)
    model = CNN(num_classes).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, val_losses = train(train_loader, val_loader, model, loss_fn, optimizer, n_epochs)
    # Calculate testing accuracy:
    predictions, labels = test(test_loader, model)
    test_accuracy = np.sum(predictions == labels) / len(labels)
    print(f'The testing accuracy is: {test_accuracy}.')
    torch.save(model, '../models/conv2d_mnist.pt')