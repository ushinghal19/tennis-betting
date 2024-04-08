import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BaseModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        for hidden_layer in self.hidden_layers:
            
            x = hidden_layer(x)
            x = self.relu(x)
        
        x = self.fc2(x)

        return x