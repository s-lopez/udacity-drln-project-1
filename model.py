import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # define the layers
        # two linears with relus, for starters
        # inputs: State_size
        # outputs: One per action (4)
        #State shape:  (8,)
        #Number of actions:  4
        self.l1 = nn.Linear(state_size, 90)
        self.l2 = nn.Linear(90, 45)
        self.l3 = nn.Linear(45, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Simple forward logic
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        
        return x
