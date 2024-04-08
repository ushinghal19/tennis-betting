import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.dataset import random_split


class TennisDataset(Dataset):
    """
    Dataset for our tennis data. The labels are represented as 1 for if player 1 won, 0 if
    player 2 won.
    """
    def __init__(self, features_df, target_array):
        # Convert the features and labels into numpy arrays if they're not already
        features_np = features_df.to_numpy().astype(np.float32)
        # Converts winners to a binary format where 1 is Player 1 won, 0 otherwise
        # labels_np = (target_array == features_df['Player 1'].values).astype(np.float32)
        labels_np = target_array.astype(np.float32)

        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features_np)
        self.labels = torch.tensor(labels_np)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


def read_data():
    directory = 'data'
    all_filenames = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])
    df_list = []

    for filename in all_filenames:
        df = pd.read_csv(filename)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


def process_data(drop_winner_rank=True):
    df = read_data()

    columns_to_keep = [
        'ATP', 'Location', 'Tournament', 'Date', 'Series', 'Winner', 'Loser',
        'Court', 'Surface', 'Round', 'Best of', 'WRank', 'LRank', 'WPts', 'LPts'
    ]

    df_selected = df[columns_to_keep].copy()

    if not drop_winner_rank: 
        return df_selected

    # Update mappings before shuffling and dropping
    df_selected['Court'] = df_selected['Court'].map({'Indoor': float(1), 'Outdoor': float(0)})
    df_selected['Surface'] = df_selected['Surface'].map({'Hard': float(3), 'Clay': float(2), 'Carpet': float(1), 'Grass': float(0)})
    df_selected['Series'] = df_selected['Series'].map({'Masters 1000': float(7), 'ATP500': float(6), 'Masters Cup': float(5), 'ATP250': float(4), 'Grand Slam': float(3), 'Masters': float(2), 'International': float(1), 'International Gold': float(0)})

    def shuffle_row(row):
        if np.random.rand() > 0.5:
            return pd.Series([row['Winner'], row['Loser'], row['WRank'], row['LRank'], row['WPts'], row['LPts']], index=['Player 1', 'Player 2', 'Rank 1', 'Rank 2', 'Pts 1', 'Pts 2'])
        else:
            return pd.Series([row['Loser'], row['Winner'], row['LRank'], row['WRank'], row['LPts'], row['WPts']], index=['Player 1', 'Player 2', 'Rank 1', 'Rank 2', 'Pts 1', 'Pts 2'])

    # Apply shuffling
    df_selected[['Player 1', 'Player 2', 'Rank 1', 'Rank 2', 'Pts 1', 'Pts 2']] = df_selected.apply(shuffle_row, axis=1)

    # Extract target array based on shuffled data
    target_array = (df_selected['Player 1'] == df_selected['Winner']).astype(float).values

    features_df = df_selected.drop(['Winner', 'Loser', "WRank", "LRank", "WPts", "LPts", 'Player 1', 'Player 2', 'Date', 'Round', 'Location', 'Tournament'], axis=1)
    return features_df, target_array


def load_dataset():
    """
    Load and return the dataset of tennis data.
    """
    pd.set_option('display.max_columns', None)
    dfs, wl = process_data()
    dataset = TennisDataset(features_df=dfs, target_array=wl)
    return dataset

# pd.set_option('display.max_columns', None)
# dfs, wl = process_data()
# print(dfs.head())
# print(wl[:5])
