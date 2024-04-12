"""
Training Module.
"""
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from read_data import TennisDataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch
from BaseModel import BaseModel
import os
import random

def accuracy(model, data_loader):
    """
    Evaluate model on a Data Loader.
    """
    model.eval()
    correct = 0

    # Disable gradient computation
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            # Run model
            output = model(batch_data).squeeze()

            # Softmax
            y = (torch.sigmoid(output) > 0.5).float()

            # Increment correct count
            correct += torch.sum(y == batch_labels).item()

    return correct / len(data_loader.dataset)

def train(train_data_loader, val_data_loader, log_interval, **kwargs):
    learning_rate = kwargs['lr']
    num_epochs = kwargs['num_epochs']
    num_layers = kwargs['num_layers']
    hidden_dim = kwargs['hidden_dim']

    model = BaseModel(input_dim=11, hidden_dim=hidden_dim, num_layers=num_layers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Make directory to store this model
    # path = os.path.join(os.path.dirname(__file__), f'/../models/lr{learning_rate}_l{num_layers}_hd{hidden_dim}')
    path = f'models/lr{learning_rate}_l{num_layers}_hd{hidden_dim}'
    os.makedirs(path, exist_ok=True)

    # Training statistics
    iters, train_loss, val_acc = [], [], []
    iter_count = 0

    try:
        for epoch in range(num_epochs):
            for i, (data, targets) in enumerate(train_data_loader):
                X = data
                t = targets

                z = model(X).squeeze()
                loss = criterion(z, t)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                iter_count += 1

                if iter_count % log_interval == 0:
                    iters.append(iter_count)
                    train_loss.append(loss.item())

                    val_acc.append(accuracy(model, val_data_loader))

                    if iter_count % log_interval * 2 == 0:
                        print(f'Train loss: {loss.item()}')
                        print(f'Validation accuracy: {val_acc[-1]}')

            # Save model every epoch
            torch.save(model.state_dict(), f'{path}/model_e{epoch + 1}.pth')
    finally:
        # Plot data even if training is interrupted
        plt.figure()
        plt.plot(iters[:len(train_loss)], train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.savefig(f'{path}/train_loss.png')

        plt.figure()
        plt.plot(iters[:len(val_acc)], val_acc)
        plt.title("Accuracy over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"])

        plt.savefig(f'{path}/accuracy.png')


def grid_search(log_interval, **kwargs):
    # Grid search
    train_loader, val_loader, test_loader = get_dataloaders()

    for num_layers in kwargs['num_layers']:
        for hidden_dim in kwargs['hidden_dim']:
            for lr in kwargs['lr']:
                hyperparams = {
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_epochs': 100,
                }
                print(f'Training hidden_dim={hidden_dim}, num_layers={num_layers}, lr={lr}')
                print('---------------------------------------------------')
                train(train_loader, val_loader, log_interval, **hyperparams)
                print('---------------------------------------------------\n')


def test_correct():
    dataset = load_dataset()
    x = 50
    dataset, _ = random_split(dataset, [x, len(dataset) - x])
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=50, shuffle=True)

    hyperparams = {
        'lr': 0.0001,
        'hidden_dim': 1000,
        'num_layers': 2,
        'num_epochs': 500,
    }

    train(train_loader, train_loader, 2, **hyperparams)

def get_dataloaders():
    dataset = load_dataset()
    total_size = len(dataset)

    train_size = int(0.7 * total_size)  # 70% of the dataset for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # 15% for test (remainder)

    # Splitting the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Creating DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader

def setup():
    grid_search_vals = {
        'lr': [0.0001, 0.001],
        'hidden_dim': [50, 100, 200, 1000],
        'num_layers': [2, 3, 5]
    }

    grid_search(1000, **grid_search_vals)


if __name__ == '__main__':
    setup()
    # test_correct()



