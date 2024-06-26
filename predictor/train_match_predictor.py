"""
Training Module.
"""
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from predictor.BaseModel import BaseModel
from predictor.read_data import TennisDataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch

import os
import random

INPUT_DIM = 14

def accuracy(model, data_loader):
    """
    Evaluate model on a Data Loader.
    """
    reset_model = model.training
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

    if reset_model:
        model.train()

    return correct / len(data_loader.dataset)

def train(train_data_loader, val_data_loader, log_interval, **kwargs):
    learning_rate = kwargs['lr']
    num_epochs = kwargs['num_epochs']
    num_layers = kwargs['num_layers']
    hidden_dim = kwargs['hidden_dim']

    model = BaseModel(input_dim=INPUT_DIM, hidden_dim=hidden_dim, num_layers=num_layers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Make directory to store this model
    # path = os.path.join(os.path.dirname(__file__), f'/../models/lr{learning_rate}_l{num_layers}_hd{hidden_dim}')
    path = f'../models/lr{learning_rate}_l{num_layers}_hd{hidden_dim}'
    os.makedirs(path, exist_ok=True)

    # Training statistics
    iters, train_loss, val_acc = [], [], []
    iter_count = 0

    model.train()

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

            # Save model every epoch
            print(f'[Epoch: {epoch + 1}] Train loss: {loss.item()}')
            print(f'[Epoch: {epoch + 1}] Validation accuracy: {accuracy(model, val_data_loader)}')
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
        plt.plot(iters[:len(val_acc)], val_acc, color='orange')
        plt.title("Validation accuracy over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")

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
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))

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

    grid_search(250, **grid_search_vals)


def eval_final_model(path_to_model, **kwargs):
    model = BaseModel(input_dim=INPUT_DIM, hidden_dim=kwargs['hidden_dim'], num_layers=kwargs['num_layers'])
    model.load_state_dict(torch.load(path_to_model))

    _, _, test_loader = get_dataloaders()
    return accuracy(model, test_loader)


def eval_baseline_model():
    class BaselineModel(nn.Module):
        """
        Always predict higher ranking player to win. Note: Higher rank corresponds to
        lower number (i.e., rank 1 player is ranked higher than rank 2 player).
        """
        def __init__(self):
            super(BaselineModel, self).__init__()
        def forward(self, X):
            # Access the 9th column (index 8) and the 10th column (index 9)
            # These represent the ranks
            rankings_p1 = X[:, 8]
            rankings_p2 = X[:, 9]

            # Compare the two columns
            predictions = (rankings_p1 <= rankings_p2).long()
            return predictions.unsqueeze(0)

    baseline_model = BaselineModel()
    _, _, test_loader = get_dataloaders()

    print(f'Performance on Test Set: {accuracy(baseline_model, test_loader)}')


if __name__ == '__main__':
    # For deterministic splitting of datasets
    torch.manual_seed(42)

    # Uncomment to test training code correctness (can overfit on small dataset)
    # test_correctness()

    # Uncomment to run grid search over many hyperparmeter choices
    # setup()
