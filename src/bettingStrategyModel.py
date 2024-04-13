import torch
import torch.nn as nn
import torch.nn.functional as F

class BettingStrategyModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim=4):
        super(BettingStrategyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        y = x
        x = self.fc1(x)
        x = self.relu(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)

        x = self.fc2(x)
        return x


class BettingStrategyAttnModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim=4, num_heads=2):
        super(BettingStrategyAttnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.fc2 = nn.Linear(hidden_dim, 3)

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

def betting_loss_function(inputs, outputs, winners):
    # inputs is a tensor of shape [batch_size, 4] with columns: [predicted prob P1, predicted prob P2, 1/implied odds P1, 1/implied odds P2]
    # outputs is a tensor of shape [batch_size, 3] containing the outputs of the network
    # winners is a tensor of shape [batch_size] with values 0 for player 1 wins, 1 for player 2 wins

    batch_size = outputs.size(0)
    
    _, predicted_ids = torch.max(outputs, dim=1)  # Find the index of the predicted maximum output for each instance in the batch

    # Extract reciprocal of implied odds for each instance and each player
    odds_p1 = inputs[:, 2]  # 1/Implied probability P1
    odds_p2 = inputs[:, 3]  # 1/Implied probability P2
    no_bet = torch.zeros(batch_size, device=outputs.device)  # No bet odds are zero

    # Gather the odds based on the winner indices
    winner_odds = torch.where(winners == 0, odds_p1, odds_p2)

    # Prepare for vectorized calculation of bet results
    predicted_odds = torch.stack([odds_p1, odds_p2, no_bet], dim=1)
    bet_results = torch.gather(predicted_odds, 1, predicted_ids.unsqueeze(1)).squeeze()

    # Calculate win/loss conditions
    win_mask = (predicted_ids == winners)
    loss_mask = (predicted_ids != winners) & (predicted_ids != 2)

    # Calculate bet results
    bet_result = torch.zeros_like(winner_odds)
    bet_result[win_mask] = bet_results[win_mask]  # win: add odds
    bet_result[loss_mask] = -bet_results[loss_mask]  # loss: subtract odds

    # Compute final loss
    loss = winner_odds - bet_result

    return loss.mean()

    



