import numpy as np
import torch

from ..util.file import *
from ..util.log import *

# Parameters for BPTT policy.
#
# input_size: int (input size)
# hidden_size: int (hidden layer size)
# output_size: int (output size)
# dir: str (directory to save the network)
# fname: str (filename to save the network)
class BPTTPolicyParams:
    def __init__(self, input_size, hidden_size, output_size, dir, fname):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dir = dir
        self.fname = fname

# Neural network policy.
#
# params: BPTTPolicyParams
class BPTTPolicy:
    def __init__(self, params):
        self.params = params
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.params.input_size, self.params.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.params.hidden_size, self.params.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.params.hidden_size, self.params.output_size),
        )

    # Predict action for a single state.
    #
    # state: np.array([state_dim])
    # return: np.array([action_dim])
    def act(self, state):
        # Step 1: Convert to torch tensor
        state = torch.tensor(state, dtype=torch.float)

        # Step 2: Get torch action
        action = self.act_torch(state)

        # Step 3: Convert to numpy array
        action = action.detach().numpy()

        return action

    # Predict actions for a list of states.
    #
    # states: np.array([n_pts, state_dim])
    # return: np.array([n_pts, action_dim])
    def act_all(self, states):
        # Step 1: Convert to torch tensor
        states = torch.tensor(states, dtype=torch.float)

        # Step 2: Get torch actions
        actions = self.act_all_torch(states)

        # Step 3: Convert to numpy array
        actions = actions.detach().numpy()

        return actions

    # Predict action for a single state using torch.
    #
    # state: torch.tensor([state_dim])
    # return: torch.tensor([action_dim])
    def act_torch(self, state):
        return self.model(state)

    # Predict actions for a list of states using torch.
    #
    # states: torch.tensor([n_pts, state_dim])
    # return: torch.tensor([n_pts, action_dim])
    def act_all_torch(self, states):
        return self.model(states)

    # Save the neural network layers.
    def save(self):
        # Step 1: Ensure the directory exists
        ensure_dir(self.params.dir)

        # Step 2: Build path
        path = '{}/{}'.format(self.params.dir, self.params.fname)

        # Step 3: Save the layers
        torch.save(self.model.state_dict(), path)

        # Step 4: Logging
        log('Saved actor to {}'.format(path), INFO)

    # Load the neural network layers.
    def load(self):
        # Step 1: Build path
        path = '{}/{}'.format(self.params.dir, self.params.fname)

        # Step 2: Load the data
        data = torch.load(path)

        # Step 3: Load model
        self.model.load_state_dict(data)

        # Step 3: Logging
        log('Loaded actor from {}'.format(path), INFO)

# Parameters for BPTT policy.
#
# action_dim: int (dimension of the action space)
# n_iters: int (the number of steps of gradient descent to take)
# learning_rate: float (the learning rate to use in gradient descent)
# gamma: float (discount factor)
# print_freq: int | None (how many iterations to print, or None if never print)
class BPTTParams:
    def __init__(self, action_dim, n_iters, learning_rate, gamma, print_freq):
        self.action_dim = action_dim
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.print_freq = print_freq

# Backpropagation through time.
#
# env: TorchEnv
# policy: BPTTPolicy
# params: BPTTParams
def bptt(env, policy, params):
    # Step 1: Set up the optimizer
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=params.learning_rate)

    # Step 2: Gradient steps
    for t in range(params.n_iters):
        # Step 2a: Get a rollout
        cum_loss = _get_rollout_torch(env, policy, False, params.gamma)

        # Step 2b: Take a gradient step
        optimizer.zero_grad()
        cum_loss.backward()
        optimizer.step()

        # Step 2c: Logging
        log('{} {}'.format(t, cum_loss.item()), INFO)

        if not params.print_freq is None and t%params.print_freq == 0:
            _get_rollout_torch(env, policy, True, params.gamma)

# Get a symbolic rollout using Torch.
#
# env: TorchEnv
# policy: BPTTPolicy
# animate: bool (whether to animate the environment)
# gamma: float (discount factor)
def _get_rollout_torch(env, policy, animate, gamma):
    # Step 1: Initialization
    state = env.reset_torch()
    cum_loss = 0.0
    discount = 1.0

    # Step 2: Get rollout
    for _ in range(env.max_steps()):
        # Step 2a: Compute action
        action = policy.act_torch(state)

        # Step 2b: Transition system
        state = env.step_torch(state, action)

        # Step 2c: Compute losss
        loss = env.loss_torch(state, action)

        # Step 2d: Increment cumulative loss
        cum_loss += discount * loss

        # Step 2e: Increment discount
        discount *= gamma

        # Step 2f: Animate
        if animate:
            env.render(state.detach().numpy(), action.detach().numpy())

    return cum_loss

