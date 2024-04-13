from torch.utils.data import DataLoader, TensorDataset
import torch

from predictor.train_match_predictor import get_dataloaders
from predictor.BaseModel import BaseModel

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


if __name__ == '__main__':
    model_path = '../models/lr0.0005_l3_hd200/model_e93.pth'
    model = BaseModel(input_dim=14, hidden_dim=200, num_layers=3)
    model.load_state_dict(torch.load(model_path))

    T, _, _, = get_transformed_dataloaders(model)
    for X, t in T:
        for i in range(len(X)):
            print(X[i])
            print(t[i])
            print('\n')

        break
