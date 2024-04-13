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

class LossWithExpectedValues(nn.Module):
    def __init__(self):
        super(LossWithExpectedValues, self).__init__()

    def forward(self, outputs, targets, bet_odds):
        """
        Parameters:
        outputs : Tensor - The logits from the neural network.
        targets : Tensor - The actual labels (0 for P1 wins, 1 for P2 wins).
        bet_odds : Tensor - The odds associated with each bet option ([odds P1, odds P2]).
        
        Returns:
        torch.Tensor - Calculated loss.
        """
        # Calculate softmax to get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Indices of the actual winning odds
        actual_odds = torch.gather(bet_odds, 1, targets.unsqueeze(1)).squeeze()
        
        # Expected return for choosing each option:
        expected_returns = torch.stack([
            bet_odds[:, 0] * probs[:, 0],  # Expected return if bet on P1
            bet_odds[:, 1] * probs[:, 1],  # Expected return if bet on P2
            torch.zeros_like(probs[:, 0])  # Expected return if no bet
        ], dim=1)
        
        # Calculate the negative of expected returns as losses
        losses = -expected_returns
        
        # Collect the loss corresponding to the chosen action
        chosen_losses = torch.gather(losses, 1, targets.unsqueeze(1)).squeeze()
        
        # Mean loss across the batch
        return chosen_losses.mean()
    

class LossFunctionSingleBet(nn.Module):
    def __init__(self):
        super(LossFunctionSingleBet, self).__init__()

    def forward(self, outputs, targets, bet_odds):
        """
        Calculate loss directly based on the decision taken.
        """
        probs = F.softmax(outputs, dim=1)
        _, predicted_ids = torch.max(probs, dim=1)  # Get the most likely bet

        # Gather the odds and the decisions
        chosen_odds = torch.gather(bet_odds, 1, predicted_ids.unsqueeze(1)).squeeze()

        # Calculate direct outcomes
        win_mask = predicted_ids == targets
        loss_mask = ~win_mask

        # Bet results
        bet_results = torch.zeros_like(chosen_odds)
        bet_results[win_mask] = chosen_odds[win_mask] - 1  # profit on wins
        bet_results[loss_mask] = -1  # loss on losses

        return -bet_results.mean()  # Negative since we minimize loss