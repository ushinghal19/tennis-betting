import pandas as pd
import os
import numpy as np

def read_data():
    directory = 'data'
    all_filenames = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])
    df_list = []

    for filename in all_filenames:
        df = pd.read_csv(filename)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


def process_data():
    """Processes data for model one, returns a dataframe of features and an array of target winners"""
    df = read_data()

    columns_to_keep = [
        'ATP', 'Location', 'Tournament', 'Date', 'Series', 'Winner', 'Loser',
        'Court', 'Surface', 'Round', 'Best of', 'WRank', 'LRank', 'WPts', 'LPts'
    ]

    df_selected = df[columns_to_keep].copy()

    def shuffle_row(row):
        if np.random.rand() > 0.5:
            return pd.Series([row['Winner'], row['Loser'], row['WRank'], row['LRank'], row['WPts'], row['LPts']])
        else:
            return pd.Series([row['Loser'], row['Winner'], row['LRank'], row['WRank'], row['LPts'], row['WPts']])

    df_selected[['Player 1', 'Player 2', 'Rank 1', 'Rank 2', 'Pts 1', 'Pts 2']] = df_selected.apply(shuffle_row, axis=1)

    target_array = df_selected['Winner'].values

    features_df = df_selected.drop(['Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts'], axis=1)

    return features_df, target_array


pd.set_option('display.max_columns', None)
dfs, wl = process_data()
print(dfs.head())
print(wl[:5])
