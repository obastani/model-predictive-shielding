import numpy as np
import torch

# Parameters
_dt = 0.05
_max_a = 5.0

# Note: Because the system is relatively simple,
# we can manually compute the region of attraction
# (RoA) for the bicycle. In particular, the LQR
# brings the bicycle to a stop as quickly as possible,
# within the acceleration bounds. Thus, a point is
# guaranteed to be stable if the LQR can bring the
# bicycle to a complete stop in one step from that
# point (and a zero velocity, zero acceleration point
# is invariant).

# Check if the given state is safe, i.e., the bicycle has
# not collided with an obstacle.
#
# state: np.array([state_dim])
# x_obstacle_0: float
# x_obstacle_1: float
# obstacle_radius: float
# return: bool
def is_bicycle_safe(state, x_obstacle_0, x_obstacle_1, obstacle_radius):
    # Step 1: Unpack values
    xb, yb, xf, yf, _, y_obstacle_0, y_obstacle_1 = state

    # Step 2: Convert to lists
    zs = [(xf, yf), (xb, yb)]
    z_obstacles = [(x_obstacle_0, y_obstacle_0), (x_obstacle_1, y_obstacle_1)]

    # Step 3: Compute safety
    for z in zs:
        for z_obstacle in z_obstacles:
            dist = np.square(z[0] - z_obstacle[0]) + np.square(z[1] - z_obstacle[1])
            if dist <= obstacle_radius * obstacle_radius:
                return False
    return True

# Safe policy for bicycle. Comes to a stop as
# quickly as possible.
class BicycleSafePolicy:
    # Get safe backup action.
    #
    # state: torch.tensor([state_dim])
    def act(self, state):
        # Step 1: Get velocity
        v = state[4]

        # Step 2: Compute optimal action
        opt_a = -v / _dt

        # Step 3: Clamp action
        a = np.clip(opt_a, -_max_a, _max_a)

        # Step 4: Build action
        return torch.tensor([a, 0.0], dtype=torch.float)

# Check if the given state is stable, i.e., the bicycle
# can safely stop without colliding with an obstacle.
#
# state: np.array([state_dim])
# x_obstacle_0: float
# x_obstacle_1: float
# obstacle_radius: float
# return: bool
def is_bicycle_stable(state, x_obstacle_0, x_obstacle_1, obstacle_radius):
    # Step 1: Get velocity
    v = state[4]

    # Step 2: Compute optimal action
    opt_a = -v / _dt

    # Step 3: Check to make sure we can stop in one step
    if np.abs(opt_a) > _max_a:
        return False

    # Step 4: Take a step
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor([opt_a, 0.0], dtype=torch.float)
    next_state = BicycleEnv().step_torch(state, action)
    next_state = next_state.detach().numpy()

    # Step 5: Check safety of next state
    return is_bicycle_safe(state, x_obstacle_0, x_obstacle_1, obstacle_radius)

# States are np.array([state_dim]), where state_dim = 5
# representing a vector
#
#   [ xb, yb, xf, yf, v ]
#
# where [ xb, yb ] are the coordinates of the back of the car,
# [ xf, yf ] are the coordinates of the front of the car,
# and v is the velocity.
#
# Actions are np.array([action_dim]), where action_dim = 2
# representing a vector
#
#   [ a, t ]
#
# where a is the acceleration and t is the steering angle.
#
# Rewards are given according to
#
#   (i)  a per time step reward penalizing large actions
#   (ii) a final step reward measuring distance to the goal,
#        which specifies the desired location of the
#        front of the car
#
class BicycleEnv:
    # Initializes the goal, time step, and weight of action portion of the loss.
    def __init__(self):
        # dynamics parameters
        self.dt = _dt
        self.max_a = _max_a
        self.max_steps_ = 200

        # action loss parameters
        self.action_weight = 0.01

        # goal loss parameters
        self.goal = np.array([1.0, 0.0])

        # obstacle loss parameters
        self.obstacle_weight = 100.0
        self.obstacle_radius = 0.05
        self.x_obstacle_0 = 0.4
        self.x_obstacle_1 = 0.7

    #
    # Gym functions
    #

    # Render the current state.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def render(self, state, action):
        print(state,
              action,
              'safe',
              self.is_safe(state),
              'stable',
              self.is_stable(state),
              self._goal_loss_torch(torch.tensor(state, dtype=torch.float)).item(),
              self._obstacle_loss_torch(torch.tensor(state, dtype=torch.float)).item(),
              self._action_loss_torch(torch.tensor(action, dtype=torch.float)).item())

    # Get the final reward.
    #
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def final_loss(self, state, action):
        return state[2]

    # Compute whether the current state is safe.
    #
    # state: np.array([state_dim])
    def is_safe(self, state):
        return is_bicycle_safe(state, self.x_obstacle_0, self.x_obstacle_1, self.obstacle_radius)

    # Compute whether the current state is stable.
    #
    # state: np.array([state_dim])
    def is_stable(self, state):
        return is_bicycle_stable(state, self.x_obstacle_0, self.x_obstacle_1, self.obstacle_radius)

    # Close the environment.
    def close(self):
        pass

    # Maximum number of steps.
    #
    # return: int
    def max_steps(self):
        return self.max_steps_
    #
    # TorchEnv functions
    #

    # Get a random initial state.
    #
    # return: np.array([state_dim])
    def reset_torch(self):
        y_obstacle_0 = 2.0 * self.obstacle_radius * np.random.uniform() - self.obstacle_radius
        y_obstacle_1 = 2.0 * self.obstacle_radius * np.random.uniform() - self.obstacle_radius
        state = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, y_obstacle_0, y_obstacle_1])
        return torch.tensor(state, dtype=torch.float)

    # Takes a single step according to the dynamics,
    # using torch tensors to enable automatic gradient computations.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([action_dim])
    # return: torch.tensor([state_dim])
    def step_torch(self, state, action):
        # Step 1: Unpack values
        dt = self.dt
        max_a = self.max_a
        ns = state.clone()
        a, t = action

        # Step 2: Threshold acceleration
        a = torch.clamp(a, -max_a, max_a)

        # Step 3: Update car velocity
        ns[4] += a * dt

        # Step 4: Update car front x and y
        XL = ns[2] - ns[0]
        YL = ns[3] - ns[1]
        # H is a constant, so we discard the gradient computation
        H = np.sqrt((XL.pow(2) + YL.pow(2)).item())
        coa = XL/H
        sia = YL/H
        DY = ns[4] * dt * (torch.sin(t) * coa + torch.cos(t) * sia)
        DX = ns[4] * dt * (torch.cos(t) * coa - torch.sin(t) * sia)

        ns[2] += DX
        ns[3] += DY

        # Step 5: Update car back x and y
        tt = (DX + XL) * coa + (DY + YL) * sia
        st = 4.0 * tt * tt - 4.0 * (DX * DX + 2.0 * DX * XL + DY * (DY + 2.0 * YL))
        q = DX * coa + XL * coa + DY * sia + YL * sia - 0.5 * torch.sqrt(st)

        ns[0] += q * coa
        ns[1] += q * sia

        return ns

    # Loss function for a given step, computed using torch
    # tensors to enable automatic gradient computation.
    #
    # state: torch.tensor([state_dim])
    # action: torch.tensor([action_dim])
    # return: torch.tensor(1)
    def loss_torch(self, state, action):
        # Step 1: Loss from distance to goal
        loss = self._goal_loss_torch(state)

        # Step 2: Loss from distance to obstacles
        loss += self._obstacle_loss_torch(state)

        # Step 3: Loss from action
        loss += self._action_loss_torch(action)

        return loss

    # Get the loss for distance to goal.
    #
    # state: torch.tensor([state_dim])
    # return: torch.tensor(1)
    def _goal_loss_torch(self, state):
        # Step 1: Unpack values
        _, _, xf, yf, _, _, _ = state
        xf_goal, yf_goal = self.goal

        # Step 2: Compute loss
        loss = (xf - xf_goal).pow(2) + (yf - yf_goal).pow(2)

        return loss

    # Get the loss for distance to obstacles.
    #
    # state: torch.tensor([state_dim])
    # return: torch.tensor(1)
    def _obstacle_loss_torch(self, state):
        # Step 1: Unpack values
        xb, yb, xf, yf, _, y_obstacle_0, y_obstacle_1 = state
        x_obstacle_0 = self.x_obstacle_0
        x_obstacle_1 = self.x_obstacle_1
        obstacle_radius = self.obstacle_radius

        # Step 2: Convert to lists
        zs = [(xf, yf), (xb, yb)]
        z_obstacles = [(x_obstacle_0, y_obstacle_0), (x_obstacle_1, y_obstacle_1)]

        # Step 3: Compute loss
        loss = 0.0
        for z in zs:
            for z_obstacle in z_obstacles:
                cur_loss = 2.0 * obstacle_radius * obstacle_radius
                cur_loss -= (z[0] - z_obstacle[0]).pow(2)
                cur_loss -= (z[1] - z_obstacle[1]).pow(2)
                cur_loss = torch.max(torch.tensor([0.0, cur_loss], dtype=torch.float))
                loss += cur_loss

        return self.obstacle_weight * loss

    # Get the loss for the action
    #
    # action: torch.tensor([action_dim])
    # return: torch.tensor(1)
    def _action_loss_torch(self, action):
        return self.action_weight * action.pow(2).sum()

# Bicycle environment for training the
# recovery policy.
class BicycleRecoveryEnv:
    # Initializes the parameters of the model.
    #
    # policy: BPTTPolicy
    def __init__(self, policy):
        # Step 1: Environment
        self.env = BicycleEnv()

        # Step 2: Policy
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
            if not self.env.is_safe(state.detach().numpy()):
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
        # Step 1: Loss from distance to goal
        loss = self._goal_loss_torch(state)

        # Step 2: Loss from distance to obstacles
        loss += self.env._obstacle_loss_torch(state)

        # Step 3: Loss from action
        loss += self.env._action_loss_torch(action)

        return loss

    # Get the loss for distance to goal, which is v = 0.
    #
    # state: torch.tensor([state_dim])
    # return: torch.tensor(1)
    def _goal_loss_torch(self, state):
        # Step 1: Unpack values
        _, _, _, _, v, _, _ = state

        # Step 2: Compute loss
        loss = v.pow(2)

        return loss

    # Maximum number of steps.
    #
    # return: int
    def max_steps(self):
        return self.env.max_steps()
