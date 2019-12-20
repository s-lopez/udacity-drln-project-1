import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# Reference to the parameters used in Prioritized Experience Replay (PER):
# [1] https://danieltakeshi.github.io/2019/07/14/per/
# [2] https://arxiv.org/pdf/1511.05952.pdf

# Define the agent's parameters as a namedtuple
Parameters = namedtuple('Parameters', ['buffer_size',       # replay buffer size
                                       'batch_size',        # minibatch size
                                       'gamma',             # discount factor
                                       'tau',               # for soft update of target parameters
                                       'lr',                # learning rate 
                                       'update_every',      # how often to update the network
                                       'state_size',        # Dimensionality of the state-space
                                       'action_size',       # Dimensionality of the action-space
                                       'use_per',           # Activate Prioritized Experience Replay?
                                       'per_min_priority',  # The minimum priority of an experience ("Epsilon" in [2])
                                       'per_prio_coeff',    # The priority exponent ("Alpha" in [2])
                                       'per_w_bias_coeff',  # The weight correction term exponent ("Beta" in [2])
                                       'seed'])

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, params, cuda=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            cuda (bool): If True, tries to use the GPU
            params (dqn_agent.Parameters): The agent's hyperparameters
        """
        
        # Unpack the agent's parameters
        (self.buffer_size, 
         self.batch_size, 
         self.gamma, 
         self.tau,
         self.lr, 
         self.update_every,
         self.state_size,
         self.action_size,
         self.use_per,
         self.per_min_priority,
         self.per_prio_coeff,
         self.per_w_bias_coeff,
         self.seed) = params
        
        random.seed(self.seed)

        # Q-Network
        if cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size,
                                   self.buffer_size,
                                   self.batch_size,
                                   self.seed,
                                   self.device,
                                   self.use_per,
                                   self.per_prio_coeff,
                                   self.per_w_bias_coeff)
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

                
    def decide(self, state, eps=0.):
        """Returns an action for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Compute action values from the state
        state = self._to_tensor(state)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).cpu().data.numpy() 
        self.qnetwork_local.train()
            
        # Follow epsilon-greedy policy
        if random.random() <= eps:
            # Random choice
            action = random.choice(np.arange(self.action_size))
        else:
            # Greedy decision
            action = np.argmax(action_values)
        
        # Get expectation: Q(s, a) - This will be used in PER later and saves computation time
        expectation = action_values[0, action]
        
        return action, expectation
    
    
    def step(self, state, action, reward, expectation, next_state, done):
        """Perform a learning step after having decided and executed an action"""
        
        # Compute the priority of the current experience
        if self.use_per:
            with torch.no_grad():
                next_state = self._to_tensor(next_state)
                maxQ = self.max_Q_target(next_state).data.cpu().numpy().squeeze()
                td_error = reward + self.gamma * maxQ - expectation
                priority = np.abs(td_error) + self.per_min_priority
        else:
            priority = None
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, priority)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
            

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Unpack
        states, actions, rewards, next_states, dones, weights = experiences
        
        # Return the maximum target Q values for the next states
        Q_targets = rewards + gamma * self.max_Q_target(next_states).unsqueeze(1) * (1 - dones)
        
        # Return the local Q values for the state-action pairs
        Q_pred = self.Q_local(states, actions)
        
        # Compute the loss and correct for PER bias if applicable
        if self.use_per:
            weights = self._to_tensor(weights)
            loss = ((Q_targets - Q_pred)**2 * weights).mean()
        else:
            loss = F.mse_loss(Q_pred, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     


    def max_Q_target(self, states):
        """Compute max(Q_target(states, a)) over a """
        # .detach: Stop tracking gradients for this tensor
        # .max(1)[0]: Get the maximum Q value over all actions
        return self.qnetwork_target(states).detach().max(1)[0]
  

    def Q_local(self, states, actions):
        """Compute max(Q_local(states, actions))"""
        # .gather(1, actions): Returns the Q values corresponding to actions
        return self.qnetwork_local(states).gather(1, actions)
        
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def _to_tensor(self, x):
        """Convert from numpy array to tensor in suitable shape and device"""
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    
    def __init__(self, action_size, buffer_size, batch_size, seed, device, use_per, per_prio_coeff, per_w_bias_coeff):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): The target torch device
            prio_exp_replay (bool): Whether to activate prioritized experience replay
            #####missing
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  
        #self.priorities = deque(maxlen=buffer_size)
        self.priorities = np.zeros(buffer_size) # Slightly faster than a deque
        self.prio_counter = 0
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
        self.use_per = use_per
        self.per_prio_coeff = per_prio_coeff
        self.per_w_bias_coeff = per_w_bias_coeff
        self.experience = namedtuple("Experience", field_names=["state", 
                                                                "action", 
                                                                "reward", 
                                                                "next_state", 
                                                                "done"])

        
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if self.use_per:
            #self.priorities.append(priority ** self.per_prio_coeff)
            if self.prio_counter < self.buffer_size:
                self.priorities[self.prio_counter] = priority ** self.per_prio_coeff
                self.prio_counter += 1
            else:
                self.priorities[:-1] = self.priorities[1:]
                self.priorities[-1] = priority ** self.per_prio_coeff

        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.use_per:
            # Compute the sampling probabilities
            #priorities = np.array([e.priority for e in self.priorities if e is not None]) ** self.per_prio_coeff
            N = len(self.memory)
            priorities = self.priorities[:N]
            #priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            
            # Sample the experiences -- Broken: Bug in numpy?
            #experiences = np.random.choice(self.memory, size=self.batch_size, p=probabilities, replace=False)
            
            # Workaround
            # Generate the indices & extract the experiences
            indeces = np.random.choice(N, size=self.batch_size, p=probabilities, replace=False)
            experiences = [self.memory[i] for i in indeces] # Probably slow
            
            # Compute the error weights. This will be used just before the gradient descent step
            weights = (1/(probabilities * N)) ** self.per_w_bias_coeff
            weights[weights > 1] = 1 # Weight cap
        else:
            # Sample at random - weighting unneccessary
            weights = None
            experiences = random.sample(self.memory, k=self.batch_size)

        # To tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones, weights)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)