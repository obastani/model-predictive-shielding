import numpy as np
import torch
import gym

# Parameters
_safe_thresh = 0.15
_stable_thresh = 0.01

# Check if the given state is in the safe region for cart-pole.
# Note that the safety threshold is chosen.
#
# state: np.array([state_dim])
# return: bool
def is_cartpole_safe(state):
    return np.abs(state[2]) <= _safe_thresh

# Check if the given state is in the region of attraction
# for the LQR controller for cart-pole around the origin.
#
# Note: For simplicity, we use a subset of the full region
# of attraction computed by the MATLAB code. In particular,
# we restrict to the region || x || <= 0.03. We do so simply
# because this region was slightly easier to evaluate. As
# discussed in ../control/shield.py, this choice is
# compatible with our implementation of the shield policy.
#
# state: np.array([state_dim])
# return: bool
def is_cartpole_stable(state):
    return np.abs(state[1]) <= _stable_thresh and np.abs(state[2]) <= _stable_thresh and np.abs(state[3]) <= _stable_thresh

# Cart-pole environment with continuous actions.
# Stabilizes towards the origin.
class CartPoleStableEnv:
    # Initializes the parameters of the model.
    #
    # goal: np.array([state_dim]) (the goal state)
    def __init__(self, goal):
        # Step 1: Reward parameters
        self.u_weight = 1e-3
        self.goal = goal

        # Step 2: Physics parameters
        self.f = 10.0
        self.g = 9.8
        self.mc = 1.0
        self.mp = 0.1
        self.l = 0.5
        self.dt = 0.02
        self.max_steps_ = 200
        #self.max_steps_ = 1000

        # Step 3: Data structures for rendering
        self.env = gym.make('CartPole-v0')
        self.env.reset()

    #
    # Gym functions
    #

    # Render the current state.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def render(self, state, action):
        self.env.env.state = state
        self.env.render()

    # Close the environment
    def close(self):
        self.env.close()

    #
    # TorchEnv functions
    #

    # Get a random state.
    #
    # return: torch.tensor([state_dim])
    def reset_torch(self):
        state = 0.1 * np.random.uniform(size=4) - 0.05
        return torch.tensor(state, dtype=torch.float)

    # Take a single step according to the dynamics,
    # using torch tensors to enable automatic gradient computations.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([1])
    # return: torch.tensor([state_dim])
    def step_torch(self, state, action):
        # Step 0: Unpack parameters
        f, g, mc, mp, l, dt = self.f, self.g, self.mc, self.mp, self.l, self.dt
        m = mc + mp
        
        # Step 1: Unpack state and action
        x, v, t, w = state
        u = action[0]

        # Step 2: Scale action
        u = f * u

        # Step 3: Intermediate values
        ct = torch.cos(t)
        st = torch.sin(t)
        tmp = (u + mp * l * w.pow(2) * st) / m

        # Step 4: Accelerations
        wp = (g * st - tmp * ct) / (l * (4.0/3.0 - mp * ct.pow(2) / m))
        vp = tmp - mp * l * wp * ct / m

        # Step 5: Update state
        # important: don't use += here!
        x = x + v * dt
        v = v + vp * dt
        t = t + w * dt
        w = w + wp * dt

        # Step 6: Record state
        new_state = state.clone()
        new_state[0] = x
        new_state[1] = v
        new_state[2] = t
        new_state[3] = w

        return new_state

    # Loss function for a given step, computed using torch
    # tensors to enable automatic gradient computation.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([action_dim])
    # return: torch.tensor(1)
    def loss_torch(self, state, action):
        goal = torch.tensor(self.goal, dtype=torch.float)
        loss = (state - goal).pow(2).sum() + self.u_weight * action.pow(2).sum()
        return loss

    # Maximum number of steps.
    #
    # return: int
    def max_steps(self):
        return self.max_steps_

# Cart-pole environment with continuous actions.
# Reward for moving to the right.
class CartPoleMoveEnv:
    # Initializes the parameters of the model.
    def __init__(self):
        # Step 1: Reward parameters
        self.u_weight = 1e-3
        self.v_target = 0.1
        self.v_weight = 1.0
        self.t_target = 0.1
        self.t_weight = 1.0

        # Step 2: Environment
        self.env = CartPoleStableEnv(np.array([0.0, 0.0, 0.0, 0.0]))

    #
    # Gym functions
    #

    # Render the current state.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def render(self, state, action):
        self.env.render(state, action)

    # Get final loss.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def final_loss(self, state, action):
        return state[0]

    # Close the environment
    def close(self):
        self.env.close()

    #
    # TorchEnv functions
    #

    # Get a random initial state.
    #
    # return: torch.tensor([state_dim])
    def reset_torch(self):
        state = self.env.reset_torch()
        return state

    # Take a single step according to the dynamics,
    # using torch tensors to enable automatic gradient computations.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([1])
    # return: torch.tensor([state_dim])
    def step_torch(self, state, action):
        new_state = self.env.step_torch(state, action)
        return new_state

    # Loss function for a given step, computed using torch
    # tensors to enable automatic gradient computation.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([action_dim])
    # return: torch.tensor(1)
    def loss_torch(self, state, action):
        # Step 1: Initialize loss
        loss = 0.0

        # Step 2: Loss from distance to targets
        loss += self.v_weight * (state[1] - self.v_target).pow(2)
        loss += self.t_weight * (state[2] - self.t_target).pow(2)

        # Step 3: Loss from action
        loss += self.u_weight * action.pow(2).sum()

        return loss

    # Maximum number of steps.
    #
    # return: int
    def max_steps(self):
        return self.env.max_steps()

# Cart-pole environment with continuous actions.
# Initial state is obtained by executing the
# given policy for a uniformly random number
# of steps.
class CartPoleRecoveryEnv:
    # Initializes the parameters of the model.
    #
    # policy: BPTTPolicy
    def __init__(self, policy):
        # Step 1: Reward parameters
        self.u_weight = 1e-3
        self.v_weight = 1.0
        self.t_weight = 1.0
        self.w_weight = 1.0
        self.max_steps_ = 200

        # Step 2: Environment
        self.env = CartPoleStableEnv(np.array([0.0, 0.0, 0.0, 0.0]))

        # Step 3: Policy
        self.policy = policy

    #
    # Gym functions
    #

    # Render the current state.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def render(self, state, action):
        self.env.render(state, action)

    # Close the environment
    def close(self):
        self.env.close()

    #
    # TorchEnv functions
    #

    # Get a random initial state. In particular,
    # execute the environment for a uniformly random
    # number of steps using the given policy, and
    # then return the final state.
    #
    # return: torch.tensor([state_dim])
    def reset_torch(self):
        # Step 1: Randomly sample number of steps to take
        steps = np.random.randint(self.env.max_steps())

        # Step 2: Randomly sample initial state
        state = self.env.reset_torch()

        # Step 3: Simulate environment for that many steps
        for _ in range(steps):
            # Step 3a: Compute action
            action = self.policy.act_torch(state)

            # Step 3b: Compute state transition
            state = self.env.step_torch(state, action)

            # Step 3c: Break if unsafe
            if not is_cartpole_safe(state.detach().numpy()):
                break

        # Step 4: Clean state
        state = torch.tensor(state.detach().numpy(), dtype=torch.float)

        return state

    # Take a single step according to the dynamics,
    # using torch tensors to enable automatic gradient computations.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([1])
    # return: torch.tensor([state_dim])
    def step_torch(self, state, action):
        new_state = self.env.step_torch(state, action)
        return new_state

    # Loss function for a given step, computed using torch
    # tensors to enable automatic gradient computation.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([action_dim])
    # return: torch.tensor(1)
    def loss_torch(self, state, action):
        # Step 1: Initialize loss
        loss = 0.0

        # Step 2: Loss from distance to targets
        loss += self.v_weight * state[1].pow(2)
        loss += self.t_weight * state[2].pow(2)
        loss += self.w_weight * state[3].pow(2)

        # Step 3: Loss from action
        loss += self.u_weight * action.pow(2).sum()

        return loss

    # Maximum number of steps.
    #
    # return: int
    def max_steps(self):
        return self.max_steps_
