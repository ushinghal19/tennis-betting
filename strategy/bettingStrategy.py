from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from predictor.train_match_predictor import get_dataloaders
from predictor.BaseModel import BaseModel
from strategy.bettingStrategyModel import BettingStrategyModel, CustomLoss
from torch.utils.data.dataset import random_split

def get_transformed_dataloaders(model):
    """
    Get the data loaders that will be used as input to the betting strategy model.
    :param model: Optimal model chosen for predicting tennis matches.
    :return: Train, val, and test data loaders for betting strategy model. Each data
             instance includes (Bet365 Odds P1, Bet365 Odds P2, Prob P1 Wins).
             The labels are (1 if P1 wins, 0 if P2 wins).
    """
    train_loader, val_loader, test_loader = get_dataloaders()

    def create_transformed_loader(loader):
        model.eval()  # Set model to evaluation mode
        features = []
        targets = []

        with torch.no_grad():
            for X, t in loader:
                output = torch.sigmoid(model(X)) # Model's output

                # Assuming X is a tensor where the last two columns are what you need
                second_last_column = X[:, -2:-1]  # Second-last column
                last_column = X[:, -1:]  # Last column

                # Combine the desired features into one tensor
                combined_features = torch.cat((second_last_column, last_column, output), dim=1)
                features.append(combined_features)

                # Collect targets
                targets.append(t)

        # Combine all batches into a single dataset for features and targets
        all_features = torch.cat(features)
        all_targets = torch.cat(targets)

        # Create a new TensorDataset with features and targets
        transformed_dataset = TensorDataset(all_features, all_targets)

        # Create a new DataLoader
        transformed_loader = DataLoader(transformed_dataset, batch_size=128, shuffle=False)
        return transformed_loader

    train_loader = create_transformed_loader(train_loader)
    val_loader = create_transformed_loader(val_loader)
    test_loader = create_transformed_loader(test_loader)

    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, debug=False):
    """
    Evaluate our model on a Data Loader. Returns a tuple containing (total profit, avg profit, number of bets made).
    """
    reset_training = model.training
    model.eval()  # Set the model to evaluation mode

    total_profit = 0.0
    num_bets = 0

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            output = model(batch_data)

            if debug:
                print(output)
                print(batch_labels)

            # Get bets to place
            # Done using indices of highest prob. outcomes (0 for p1, 1 for p2, 2 for no bet)
            bets = torch.argmax(output, dim=1)

            if debug:
                print(bets)

            for i in range(len(bets)):
                actual_winner = int(batch_labels[i].item())
                # P1 wins -> actual_winner = 1, but idx = 0
                # P2 wins -> actual_winner = 0, but idx = 1
                winner_prob_idx = 1 - actual_winner

                chosen_bet = bets[i].item()

                if debug:
                    print(f'Bet on {chosen_bet}, winner = {winner_prob_idx} (actual_winner = {actual_winner})')
                    print(f'Odds P1: {batch_data[i, 0]}, Odds P2: {batch_data[i, 1]}')

                # Skip bets where model chooses not to bet
                if chosen_bet == 2:
                    continue

                # Check if we bet on the winner
                if chosen_bet == winner_prob_idx:
                    # batch_data[i, winner_prob_idx] = Bet365 odds for winner
                    # Profit = Stake x (Odds - 1) = Odds - 1 ; (stake = $1)
                    total_profit += batch_data[i, winner_prob_idx].item() - 1
                    if debug:
                        print(f'Profit changed by: {batch_data[i, winner_prob_idx].item() - 1}')
                # Otherwise, we lost our stake ($1)
                else:
                    total_profit -= 1
                    if debug:
                        print(f'Profit changed by: -1')

                num_bets += 1

    avg_profit = total_profit / num_bets if num_bets > 0 else 0

    if reset_training:
        model.train()

    if debug:
        print(f'Final: {total_profit, avg_profit, num_bets}')

    return total_profit, avg_profit, num_bets


def train(train_data_loader, val_data_loader, log_interval, **kwargs):
    learning_rate = kwargs['lr']
    num_epochs = kwargs['num_epochs']
    num_layers = kwargs['num_layers']
    hidden_dim = kwargs['hidden_dim']
    dropout = kwargs['dropout']

    model = BettingStrategyModel(input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    total_profit, avg_profit, num_bets = evaluate(model, val_data_loader)
    print(f'[Initial] Total profit: {total_profit}')
    print(f'[Initial] Average profit: {avg_profit}')
    print(f'[Initial] Bets placed: {num_bets} / {len(val_data_loader.dataset)}\n')

    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Make directory to store this model
    path = f'../models/betting/lr{learning_rate}_l{num_layers}_hd{hidden_dim}_d{dropout}'
    os.makedirs(path, exist_ok=True)
    stats_file = open(f'{path}/stats.txt', 'w')

    # Training statistics
    iters, train_loss, total_profit_lst, avg_profit_lst, num_bets_lst = [], [], [], [], []
    iter_count = 0

    model.train()

    try:
        for epoch in range(num_epochs):
            for i, (data, targets) in enumerate(train_data_loader):
                X = data
                t = targets

                z = model(X)
                # X[:, 0] = odds for p1, X[:, 1] = odds for p2, X[:, 2] = prob p1 wins
                loss = criterion(z, t, X[:, 0].clone().detach(), X[:, 1].clone().detach(), X[:, 2].clone().detach())

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if iter_count % log_interval == 0:
                    iters.append(iter_count + 1)
                    train_loss.append(loss.item())
                    total_profit, avg_profit, num_bets = evaluate(model, val_data_loader)
                    total_profit_lst.append(total_profit)
                    avg_profit_lst.append(avg_profit)
                    num_bets_lst.append(num_bets)

                iter_count += 1

            # Save model every epoch
            print(f'[Epoch: {epoch + 1}] Train loss: {loss.item()}')
            total_profit, avg_profit, num_bets = evaluate(model, val_data_loader)
            print(f'[Epoch: {epoch + 1}] Total profit: {total_profit}')
            print(f'[Epoch: {epoch + 1}] Average profit: {avg_profit}')
            print(f'[Epoch: {epoch + 1}] Bets placed: {num_bets} / {len(val_data_loader.dataset)}\n')

            stats_file.write(f'[Epoch: {epoch + 1}] Train loss: {loss.item()}')
            stats_file.write(f'[Epoch: {epoch + 1}] Total profit: {total_profit}')
            stats_file.write(f'[Epoch: {epoch + 1}] Average profit: {avg_profit}')
            stats_file.write(f'[Epoch: {epoch + 1}] Bets placed: {num_bets} / {len(val_data_loader.dataset)}\n')
            torch.save(model.state_dict(), f'{path}/model_e{epoch + 1}.pth')
    finally:
        # Plot data and close files even if training is interrupted
        stats_file.close()

        plt.figure()
        plt.plot(iters[:len(train_loss)], train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.savefig(f'{path}/train_loss.png')

        plt.figure()
        plt.plot(iters[:len(total_profit_lst)], total_profit_lst, color='orange')
        plt.title("Total profit over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Profit")

        plt.savefig(f'{path}/total_profit.png')

        plt.figure()
        plt.plot(iters[:len(avg_profit_lst)], avg_profit_lst, color='orange')
        plt.title("Average profit over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Profit")

        plt.savefig(f'{path}/avg_profit.png')

        plt.figure()
        plt.plot(iters[:len(num_bets_lst)], num_bets_lst, color='green')
        plt.title("Number of bets made over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Number of bets made")

        plt.savefig(f'{path}/num_bets_made.png')


def test_correct(predictor_model):
    dataset = get_transformed_dataloaders(predictor_model)[0].dataset
    x = 50
    dataset, _ = random_split(dataset, [x, len(dataset) - x], generator=torch.Generator().manual_seed(42))
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=x, shuffle=True)

    hyperparams = {
        'lr': 0.001,
        'hidden_dim': 1000,
        'num_layers': 2,
        'num_epochs': 300,
        'dropout': 0.0
    }

    train(train_loader, train_loader, log_interval=2, **hyperparams)


def grid_search(log_interval, **kwargs):
    # Grid search
    model_path = '../models/lr0.0005_l3_hd200/model_e93.pth'
    predictor_model = BaseModel(input_dim=14, hidden_dim=200, num_layers=3)
    predictor_model.load_state_dict(torch.load(model_path))

    train_loader, val_loader, test_loader, = get_transformed_dataloaders(predictor_model)

    for num_layers in kwargs['num_layers']:
        for hidden_dim in kwargs['hidden_dim']:
            for lr in kwargs['lr']:
                for dropout in kwargs['dropout']:
                    hyperparams = {
                        'lr': lr,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'num_epochs': 50,
                        'dropout': dropout,
                    }
                    print(f'Training hidden_dim={hidden_dim}, num_layers={num_layers}, lr={lr}')
                    print('---------------------------------------------------')
                    train(train_loader, val_loader, log_interval=log_interval, **hyperparams)
                    print('---------------------------------------------------\n')


def get_predictor_model():
    model_path = '../models/lr0.0005_l3_hd200/model_e93.pth'
    predictor_model = BaseModel(input_dim=14, hidden_dim=200, num_layers=3)
    predictor_model.load_state_dict(torch.load(model_path))
    return predictor_model


def eval_baseline_model():
    class BaselineStrategyModel(nn.Module):
        """
        Always bet on player with lowest Bet365 odds.
        """
        def __init__(self):
            super(BaselineStrategyModel, self).__init__()
        def forward(self, X):
            # Batch size
            N = X.size()[0]

            # Access the 1st column (index 0) and the th column (index 9)
            # These represent the ranks
            odds_p1 = X[:, 0]
            odds_p2 = X[:, 1]

            # Initialize the result tensor
            predictions = torch.zeros((N, 3), dtype=torch.long)

            # Set [1, 0, 0] where odds_p1 < odds_p2, else [0, 1, 0]
            predictions[:, 0] = (odds_p1 < odds_p2)
            predictions[:, 1] = (odds_p1 >= odds_p2)

            return predictions

    baseline_model = BaselineStrategyModel()
    predictor = get_predictor_model()
    _, val_loader, test_loader = get_transformed_dataloaders(predictor)

    print(f'Performance on Test Set: {evaluate(baseline_model, test_loader)}')


def eval_test_model(path_to_model, **kwargs):
    model = BettingStrategyModel(input_dim=3, hidden_dim=kwargs['hidden_dim'], num_layers=kwargs['num_layers'])
    model.load_state_dict(torch.load(path_to_model))

    _, _, test_loader = get_dataloaders()
    return evaluate(model, test_loader)


if __name__ == '__main__':
    # grid_search_vals = {
    #     'lr': [0.0001, 0.001],
    #     'hidden_dim': [25, 50, 100, 200, 500],
    #     'num_layers': [2, 3, 4, 5, 7],
    #     'dropout': [0.0, 0.2, 0.5],
    # }
    #
    # grid_search(250, **grid_search_vals)

    eval_baseline_model()
