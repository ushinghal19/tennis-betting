import torch
import torch.nn as nn
import torch.nn.functional as F


class BettingStrategyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim=3, dropout=0.0):
        super(BettingStrategyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.use_dropout = dropout > 0.0
        self.dropout = nn.Dropout(dropout)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            x = self.relu(x)

            # Do not use dropout right before fc2: dropout right before last layer
            # can hurt performance: https://stats.stackexchange.com/questions/299292/dropout-makes-performance-worse
            if self.use_dropout and i < len(self.hidden_layers) - 1:
                x = self.dropout(x)

        x = self.fc2(x)
        return x

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets, odds_p1, odds_p2, prob_p1_win):
        """
        Calculate loss directly based on the decision taken.
        """
        # Apply softmax to get probabilities for each class, using temperature
        # since we want this to behave more like argmax
        T = 0.5
        probs = F.softmax(outputs / T, dim=1)

        # Calculate expected gain:
        #   - If bet on P1 and P1 wins, profit = (odds_p1 - 1) * probability of betting on P1
        #   - If bet on P2 and P2 wins, profit = (odds_p2 - 1) * probability of betting on P2
        #   - If the model learns to bet on no bet, then probs[i, 0] = probs[i, 1] = 0 and there's no financial impact
        # Recall: probs[:, 0] -> bet on p1, target = 1 -> p1 won
        #         probs[:, 1] -> bet on p2, target = 0 -> p2 won
        expected_gain_p1 = probs[:, 0] * (targets == 1).float() * (odds_p1 - 1)
        expected_gain_p2 = probs[:, 1] * (targets == 0).float() * (odds_p2 - 1)
        expected_gain = expected_gain_p1 + expected_gain_p2

        # Calculate expected losses in a similar manner
        expected_loss_p1 = probs[:, 0] * (targets == 1).float() * (-1.0)
        expected_loss_p2 = probs[:, 1] * (targets == 0).float() * (-1.0)
        expected_loss = expected_loss_p1 + expected_loss_p2

        # E[Profit] = E[Gain] + E[Loss] (note Loss <= 0)
        expected_profit = expected_gain + expected_loss

        # Calculate total profit that was possible
        total_possible_profit = ((targets == 1).float() * (odds_p1 - 1)) + ((targets == 0).float() * (odds_p2 - 1))

        # Penalty for not betting when there's an obvious winner
        obvious_winner_penalty_p1 = torch.where(
            odds_p1 > 4,
            (1 - probs[:, 1]) * odds_p1,  # Apply penalty for not betting on p2 here, since p1 will likely lose
            torch.zeros_like(probs[:, 2])
        ).sum()
        obvious_winner_penalty_p2 = torch.where(
            odds_p2 > 4,
            (1 - probs[:, 0]) * odds_p2,  # Apply penalty for not betting on p1 here, since p2 will likely lose
            torch.zeros_like(probs[:, 2])
        ).sum()
        obvious_winner_penalty = obvious_winner_penalty_p1 + obvious_winner_penalty_p2

        # Penalty for betting when there's a lot of discrepency in who wins
        # (Inversely) scale penalty by bookmaker odds because if our model is confident
        # that a match is a toss-up but the bookmaker thinks someone will win, we should
        # bet on the other person (this is a bit risky, but could pan out well).
        # Note: Numerical stability is fine: odds_p1, odds_p2 >= 1
        # Note: 2 / ... is not arbitrary. We use 2 because if odds are similar
        # 2 / (odds_p1 + odds_p2) is approx 1, but if not, it's < 1.
        no_clear_winner_penalty = torch.where(
            (prob_p1_win < 0.6) & (prob_p1_win > 0.4),
            (1 - probs[:, 2]) * (1 + 2 / (odds_p1 + odds_p2)),
            torch.zeros_like(probs[:, 2])
        ).sum()

        a = F.mse_loss(expected_profit, total_possible_profit)
        e = expected_profit.mean()
        return a - e + obvious_winner_penalty + 5 * no_clear_winner_penalty
