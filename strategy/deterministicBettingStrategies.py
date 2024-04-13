import numpy as np

def test_strategy_delta(dataset, delta=0.01):
    """
    We assume the data set is a list of tuples: [(B365 Odds 1, B365 Odds 2, Predicted P1, Predicted P2, Winner(0 or 1) )]
    """
    total_bets = 0
    total_profit = 0  # Initialize total profit

    for odd1, odd2, prob1, prob2, winner in dataset:
        implied_1, implied_2 = 1 / odd1, 1 / odd2

        if prob1 - delta > implied_1:
            total_bets += 1
            total_profit += odd1 - 1 if winner == 0 else -1
        elif prob2 - delta > implied_2:
            total_bets += 1
            total_profit += odd2 - 1 if winner == 1 else -1
    
    avg_profit = total_profit / total_bets if total_bets else 0



    if total_bets:
        print("Using delta =", delta, ", we completed a total of: ", total_bets, ", with an average profit of ", round(avg_profit*100, 3), "%")
    else:
        print("Using delta =", delta, " no bets were completed.")
    
    return avg_profit, total_profit, total_bets


def test_delta_range(data, start=0.01, end=0.5, step=0.01):
    """

    """
    deltas = np.arange(start, end + step, step)
    deltas = list(deltas.round(2))

    max_average_profit = -float('inf')
    delta_max_avg_profit = None
    num_bets_max_avg_profit = None

    max_total_profit = -float('inf')
    delta_max_total_profit = None
    num_bets_max_total_profit = None

    for delta in deltas:
        avg_profit, total_profit, total_bets = test_strategy_delta(data, delta)

        if avg_profit > max_average_profit:
            max_average_profit, delta_max_avg_profit, num_bets_max_avg_profit = avg_profit, delta, total_bets

        if total_profit > max_total_profit:
            max_total_profit, delta_max_total_profit, num_bets_max_total_profit = total_profit, delta, total_bets
    
    print("The delta with the highest average return is: ", delta_max_avg_profit, " which completes ", num_bets_max_avg_profit, " bets and generates on average ", round(max_average_profit*100, 2), "% profit")
    print("The delta with the highest total return is: ", delta_max_total_profit, " which completes ", num_bets_max_total_profit, " bets and generates in total ", round(max_total_profit, 2), "$ of profit")

    return


start = 0.01
end = 0.5

data = None # add function
