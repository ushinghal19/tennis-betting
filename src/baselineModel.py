from read_data import process_data

"""
Here we find what the accuracy is when we just predict that the higher ranked player is going to win.
"""

def find_higher_rank_accuracy():

    dfs = process_data(drop_winner_rank=False)
    print(dfs)
    total, correct = 0,0
    i = 0
    for rank1, rank2 in zip(dfs['WRank'], dfs['LRank']):
        
        if float(rank1) < float(rank2):
            correct += 1
            
        total += 1
    return correct / total

accuracy = find_higher_rank_accuracy()
print("accuracy: ", accuracy)
    

