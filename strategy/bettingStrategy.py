from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

from predictor.train_match_predictor import get_dataloaders
from predictor.BaseModel import BaseModel
from strategy.bettingStrategyModel import BettingStrategyModel

def get_transformed_dataloaders(model):
    """
    Get the data loaders that will be used as input to the betting strategy model.
    :param model: Optimal model chosen for predicting tennis matches.
    :return: Train, val, and test data loaders for betting strategy model. Each data
             instance includes (Bet365 Odds P1, Bet365 Odds P2, Prob P1 Wins, Prob P2 Wins).
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
                combined_features = torch.cat((second_last_column, last_column, output, 1 - output), dim=1)
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


def evaluate(model, data_loader):
    """
    Evaluate our model on a Data Loader. Returns a tuple containing (total profit, avg profit, number of bets made).
    """
    reset_training = model.training
    model.eval()  # Set the model to evaluation mode

    total_profit = 0.0
    number_of_bets = 0

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            output = model(batch_data)
            probabilities = F.softmax(output, dim=1)  # Apply softmax to convert outputs to probabilities

            # Get bets to place
            # Done using indices of highest prob. outcomes (0 for p1, 1 for p2, 2 for no bet)
            bets = torch.argmax(probabilities, dim=1)

            for i in range(len(bets)):
                actual_winner = int(batch_labels[i].item())
                # P1 wins -> actual_winner = 1, but idx = 0
                # P2 wins -> actual_winner = 0, but idx = 1
                winner_prob_idx = 1 - actual_winner

                chosen_bet = bets[i].item()

                # Skip bets where model chooses not to bet
                if chosen_bet == 2:
                    continue

                # Check if we bet on the winner
                if chosen_bet == winner_prob_idx:
                    # batch_data[i, winner_prob_idx] = Bet365 odds for winner
                    # Profit = Stake x (Odds - 1)
                    total_profit += batch_data[i, winner_prob_idx].item() - 1
                # Otherwise, we lost our stake ($1)
                else:
                    total_profit -= 1

                number_of_bets += 1

    avg_profit = total_profit / number_of_bets if number_of_bets > 0 else 0

    if reset_training:
        model.train()

    return total_profit, avg_profit, number_of_bets


if __name__ == '__main__':
    model_path = '../models/lr0.0005_l3_hd200/model_e93.pth'
    predictor_model = BaseModel(input_dim=14, hidden_dim=200, num_layers=3)
    predictor_model.load_state_dict(torch.load(model_path))

    train_loader, val_loader, test_loader, = get_transformed_dataloaders(predictor_model)
    betting_model = BettingStrategyModel(hidden_dim=50, num_layers=2)

    print(evaluate(betting_model, val_loader))