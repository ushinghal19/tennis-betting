import torch
import torch.nn as nn
import torch.nn.functional as F


class BettingStrategyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim=4, dropout=False):
        super(BettingStrategyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.25)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
            if self.use_dropout:
                x = self.dropout(x)

        x = self.fc2(x)
        return x


class BettingStrategyAttnModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim=4, num_heads=2, dropout=False):
        super(BettingStrategyAttnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.fc2 = nn.Linear(hidden_dim, 3)

        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.25)

        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer normalization for stability

    def forward(self, x):
        # Initial feed-forward layer
        x = self.fc1(x)
        x = self.relu(x)

        # Processing through hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
            if self.use_dropout:
                x = self.dropout(x)

        # Prepare x for attention (requires shape [seq_len, batch, features])
        # Here we treat the last dimension of x as seq_len for attention.
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, batch, features]

        # Applying self-attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output + x  # Residual connection
        x = self.layer_norm(x)

        # Reverting shape back to [batch, features] for the final layer
        x = x.squeeze(1).transpose(0, 1)

        # Final layer
        x = self.fc2(x)
        return x


# Example of creating the model
# model = BettingStrategyAttnModel(hidden_dim=128, num_layers=2, input_dim=4, num_heads=2)
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossWithExpectedValues(nn.Module):
    def __init__(self):
        super(LossWithExpectedValues, self).__init__()

    def forward(self, inputs, targets):
        """
        Parameters:
        inputs : Tensor - The logits from the neural network; each row should contain (odds_p1, odds_p2, prob_p1, prob_p2)
        targets : Tensor - The actual labels (1 for P1 wins, 0 for P2 wins).
        
        Returns:
        torch.Tensor - Calculated loss.
        """
        odds_p1 = inputs[:, 0]
        odds_p2 = inputs[:, 1]
        probs_p1 = inputs[:, 2]
        probs_p2 = inputs[:, 3]
        
        # Expected return for choosing each option:
        expected_returns = torch.stack([
            odds_p1 * probs_p1,  # Expected return if bet on P1
            odds_p2 * probs_p2,  # Expected return if bet on P2
            torch.zeros_like(probs_p1)  # Expected return if no bet
        ], dim=1)
        
        # Negative expected returns as losses
        losses = -expected_returns
        
        # Actual odds based on the winner
        actual_odds = torch.where(targets == 1, odds_p1, odds_p2)
        
        # Collect the loss corresponding to the actual outcome
        chosen_losses = torch.gather(losses, 1, targets.unsqueeze(1)).squeeze()
        
        # Mean loss across the batch
        return chosen_losses.mean()

class LossFunctionSingleBet(nn.Module):
    def __init__(self):
        super(LossFunctionSingleBet, self).__init__()

    def forward(self, outputs, targets, odds_p1, odds_p2):
        """
        Calculate loss directly based on the decision taken.
        """
        # Calculate softmax for decision probabilities between P1 and P2
        probs = F.softmax(outputs, dim=1)
        
        # Get the most likely bet
        predicted_ids = torch.argmax(probs, dim=1)
        
        # Gather the odds and the decisions
        chosen_odds = torch.where(predicted_ids == 0, odds_p1, odds_p2)
        
        # Calculate direct outcomes
        # Recall: Predicted_id of 0 and target of 1 -> win, and pred_id of 1 and target of 0 -> win
        win_mask = predicted_ids == (1 - targets)
        loss_mask = predicted_ids == targets
        no_bet_mask = predicted_ids == 2

        # Bet results
        bet_results = torch.zeros_like(chosen_odds)
        bet_results[win_mask] = chosen_odds[win_mask] - 1  # profit on wins
        bet_results[loss_mask] = -1  # loss on losses
        bet_results[no_bet_mask] = 0

        # Negative since we minimize loss
        return -bet_results.mean()


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets, odds_p1, odds_p2):
        """
        Calculate loss directly based on the decision taken.
        """
        # Apply softmax to get probabilities for each class
        probs = F.softmax(outputs, dim=1)

        # Calculate expected profits when correct:
        #   - If bet on P1 and P1 wins, profit = (odds_p1 - 1) * probability of betting on P1
        #   - If bet on P2 and P2 wins, profit = (odds_p2 - 1) * probability of betting on P2
        #   - If the model learns to bet on no bet, then probs[i, 0] = probs[i, 1] = 0 and there's no financial impact
        # Recall: probs[:, 0] -> bet on p1, target = 1 -> p1 won
        #         probs[:, 1] -> bet on p2, target = 0 -> p2 won
        expected_profits = (probs[:, 0] * (targets == 1).float() * (odds_p1 - 1)) + (probs[:, 1] * (targets == 0).float() * (odds_p2 - 1))
        
        # Calculate total profit that was possible
        total_possible_profit = ((targets == 1).float() * (odds_p1 - 1)) + ((targets == 0).float() * (odds_p2 - 1))

        # Calculate incurred losses when wrong:
        # #   - Loss is modeled as -1 times the probability of having made the wrong decision
        # losses = -1.0 * ((probs[:, 0] * (targets == 0).float()) + (probs[:, 1] * (targets == 1).float()))
        # 
        # # Sum the expected profits and the incurred losses
        # bet_results = expected_profits + losses

        # # Return the negative mean of results since we minimize the loss
        # return -bet_results.mean()

        # Return MSE loss between total possible profit and expected model profits
        return F.mse_loss(expected_profits, total_possible_profit)
