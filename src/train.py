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
            output = model(batch_data)

            # Softmax
            y = (torch.sigmoid(output) > 0.5).float()

            # Increment correct count
            correct += torch.sum(y == batch_labels).item()

    return correct / len(data_loader.dataset)

def train(model, train_data_loader, val_data_loader, log_interval):
    learning_rate = 0.01
    num_epochs = 10

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training statistics
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0

    try:
        for _ in range(num_epochs):
            for i, (data, targets) in enumerate(train_data_loader):
                # TODO: Modify shapes and types depending on how we structured data and targets
                X = data
                t = targets

                z = model(X)
                loss = criterion(z, t)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                iter_count += 1

                if iter_count % log_interval == 0:
                    iters.append(iter_count)
                    train_loss.append(loss.item())
                    # TODO: Maybe pass data not data_loader
                    train_acc.append(accuracy(model, train_data_loader))
                    val_acc.append(accuracy(model, val_data_loader))
    finally:
        # Plot data even if training is interrupted
        plt.figure()
        plt.plot(iters[:len(train_loss)], train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.figure()
        plt.plot(iters[:len(train_acc)], train_acc)
        plt.plot(iters[:len(val_acc)], val_acc)
        plt.title("Accuracy over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"])


def test_correct():
    dataset = load_dataset()
    total_size = len(dataset)
    print(dataset[:100])
    train_loader = DataLoader(dataset[:100], batch_size=64, shuffle=True)

    model = BaseModel(input_dim=9, hidden_dim=50, num_layers=3)

    train(model, train_loader, train_loader, 2)


def setup():
    dataset = load_dataset()
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% of the dataset for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # The rest for testing, to ensure all data is used

    # Splitting the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Creating DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create model
    model = BaseModel(input_dim=9, hidden_dim=50, num_layers=3)

    print(accuracy(model, val_loader))


if __name__ == '__main__':
    #setup()
    test_correct()

