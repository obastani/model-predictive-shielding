import torch

# Wraps around a BPTT environment to produce a gym environment.
# See bptt.py for a description of functions that a BPTT environment (BPTTEnv) implements.
# In addition, wrapped environments should implement the following functions:
#
#    render: np.array([state_dim]), np.array([action_dim]) -> ()
#    close: () -> ()
#
class BPTTWrapperEnv:
    # Initialize the data structure.
    #
    # env: BPTTEnv
    def __init__(self, env):
        self.env = env

    # Construct the initial state.
    #
    # return: np.array([state_dim])
    def reset(self):
        return self.env.reset_torch().detach().numpy()

    # Take a single step according to the dynamics.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    # return: np.array([state_dim])
    def step(self, state, action):
        # Step 1: Convert state and action to torch tensors
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)

        # Step 2: Update the state
        next_state = self.env.step_torch(state, action)

        # Step 3: Convert to numpy array
        next_state = next_state.detach().numpy()

        return next_state

    # Take a single step according to the dynamics.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    # return: float
    def loss(self, state, action):
        # Step 1: Convert state and action to torch tensors
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)

        # Step 2: Get the loss for the current step against environment goal
        loss = self.env.loss_torch(state, action)

        # Step 3: Convert to numpy array
        loss = loss.detach().numpy()

        return loss

    # Get the final loss
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    # return: float
    def final_loss(self, state, action):
        return self.env.final_loss(state, action)

    # Return the maximum number of steps.
    #
    # return: int
    def max_steps(self):
        return self.env.max_steps()

    # Render the current state-action pair.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def render(self, state, action):
        self.env.render(state, action)

    # Close the environment.
    def close(self):
        self.env.close()
