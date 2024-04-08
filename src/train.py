"""
Training Module.
"""
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def accuracy(model, data_loader):
    return -1

def train(model, train_data_loader, val_data_loader, log_interval):
    learning_rate = 0.01
    num_epochs = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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