import pandas as pd
import os

def read_data():
    directory = 'data'
    all_filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    df_list = []

    print(all_filenames)

    for filename in all_filenames:
        df = pd.read_csv(filename)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df
