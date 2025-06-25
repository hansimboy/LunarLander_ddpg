import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
  def __init__(self, obs_dim, action_dim, hidden_state=(64, 32)):
    super().__init__()
    self.MLP = nn.Sequential(
        nn.Linear(obs_dim + action_dim, hidden_state[0]),
        nn.ReLU(),
        nn.Linear(hidden_state[0], hidden_state[1]),
        nn.ReLU(),
        nn.Linear(hidden_state[1], 1)
    )


  def forward(self, obs, action):
    x = torch.cat([obs, action], dim=-1)
    x = self.MLP(x)
    return x.squeeze(-1) #(batch)

class ActorNetwork(nn.Module):
  def __init__(self, obs_dim, hidden_state=(64,32)):
    super().__init__()
    self.MLP = nn.Sequential(
        nn.Linear(obs_dim, hidden_state[0]),
        nn.ReLU(),
        nn.Linear(hidden_state[0], hidden_state[1]),
        nn.ReLU(),
        nn.Linear(hidden_state[1], 2),
        nn.Tanh()
    )

  def forward(self, obs):
    x = self.MLP(obs)
    return x # (batch, 2)