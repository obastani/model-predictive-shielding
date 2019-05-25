import numpy as np
import torch

# Given a goal, construct the torch loss function
# that maps the current state to the distance to that state
# plus the norm of the action (weighted by the given value).
#
# goal: np.array([state_dim])
# u_weight: float
# return: torch.tensor([state_dim]) -> torch.tensor(1)
def get_torch_loss_for_goal(goal, u_weight):
    return lambda state, action: (state - torch.tensor(goal, dtype=torch.float)).pow(2).sum() + u_weight * action.pow(2).sum()

# Linear dynamics.
class LQREnv:
    # Initialize the environment.
    #
    # A: np.array([state_dim, state_dim])
    # B: np.array([state_dim, action_dim])
    # Q: np.array([state_dim, state_dim])
    # R: np.array([action_dim, action_dim])
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

# Represents an LQR policy u = K * x.
class LQRPolicy:
    # Initialize the policy.
    #
    # K: np.array([state_dim, action_dim])
    def __init__(self, K):
        self.K = K

    # Get the action for the given state.
    #
    # state: np.array([state_dim])
    # return: np.array([action_dim])
    def act(self, state):
        return np.matmul(self.K, state)

# Parameters for LQR.
#
# n_iters: int
# dt: float
class LQRParams:
    def __init__(self, n_iters):
        self.n_iters = n_iters

# LQR policy over normalized states and actions, i.e.,
#
#   x~ = x - x0
#   u~ = u - u0
#
class LQRNormPolicy:
    # LQR policy taking into account normalization
    # of states and actions by the given state and action.
    #
    # K: np.array([action_dim + 1, state_dim + 1]
    # state: np.array([state_dim])
    # action: np.array([action_dim])
    def __init__(self, K, state, action):
        self.K = K
        self.state = state
        self.action = action

    # Get the action u for the given state s using the formula
    #
    #   u = u0 + K' * [[x - x0], [1]]
    #   K' = K with last row dropped
    #
    # state: np.array([state_dim])
    # return: np.array([action_dim])
    def act(self, state):
        # Step 1: Compute x~
        state = np.concatenate([state - self.state, [1]])

        # Step 2: Compute u~ = K' * x~
        u = np.matmul(self.K[:-1], state)

        # Step 3: Compute u0 + K' * x~
        u = self.action + u

        return u

# Linearize the dynamics and loss around the given state
# and action, and then run LQR.
#
# env: LQREnv
# state: np.array([state_dim])
# action: np.array([action_dim])
# params: LQRParams
def lqr_nonlinear(env, state, action, params):
    # Step 1: Extract LQR environment
    lqr_env = _get_lqr_env(env, state, action)

    # Step 2: Compute LQR controller
    policy = lqr(lqr_env, params)

    # Step 3: Compute normalized LQR controller
    norm_policy = LQRNormPolicy(policy.K, state, action)

    return norm_policy

# Run LQR. A LQREnv should have the constants
#
#   A:  np.array([state_dim, state_dim]) (the dynamics f(x, u) = A x + B u)
#   B:  np.array([state_dim, action_dim]) (the dynamics f(x, u) = A x + B u)
#   Q:  np.array([state_dim, state_dim]) (the costs c(x, u) = x' Q x + u' R u)
#   R:  np.array([action_dim, action_dim]) (the costs c(x, u) = x' Q x + u' R u)
#
# env: LQREnv
# params: LQRParams
def lqr(env, params):
    # Step 1: Extract parameters
    A, B, Q, R = env.A, env.B, env.Q, env.R
    state_dim, action_dim = B.shape

    # Step 3: Initialize cost-to-go
    J = np.zeros([state_dim, state_dim])

    # Step 4: LQR iterations
    for _ in range(params.n_iters):
        # Step 4a: Compute K
        BJ = np.matmul(B.T, J)
        a = R + np.matmul(BJ, B)
        b = - np.matmul(BJ, A)
        K = np.matmul(np.linalg.pinv(a), b)

        # Step 4b: Update J
        ApBK = A + np.matmul(B, K)
        J = Q + np.matmul(np.matmul(K.T, R), K) + np.matmul(np.matmul(ApBK.T, J), ApBK)

    return LQRPolicy(K)

# Compute the LQR environment for the given BPTT environment
# by linearizing the dynamics around the given state and action.
#
# Note that we have the expansion
#
#   f(x, u) = f(x0, u0)
#             + fx(x0, u0) * (x - x0)
#             + fu(x0, u0) * (u - u0)
#             + O((x - x0)^2) + O((u - u0)^2)
#
# Then, we take
#
#   x~ = [x - x0, 1]
#   u~ = [u - u0, 1]
#
# and
#
#   A = [[fx(x, u), f(x, u) - x0],
#        [0,        1           ]]
#
#   B = [[fu(x, u), 0],
#        [0,        0]]
#
# so
#
#   x~' = f(x, u) - x0 = A x~ + B u~ + O(x^2) + O(u^2)
#
# Similarly, assuming we have the expansion
#
#   c(x, u) = c(x0, u0)
#             + cx(x0, u0) * (x - x0)
#             + cu(x0, u0) * (u - u0)
#             + 0.5 * (x - x0) * cxx(x0, u0) * (x - x0)
#             + 0.5 * (u - u0) * cuu(x0, u0) * (u - u0)
#             + O(dx^3) + O(du^3)
#
# assuming cxu(x0, u0) = 0. Then, we take
#
#   Q = [[0.5 * cxx(x0, u0), 0.5 * cx(x0, u0)],
#        [0.5 * cx(x0, u0),  c(x0, u0)       ]]
#
#   R = [[0.5 * cuu(x0, u0), 0.5 * cu(x0, u0)],
#        [0.5 * cu(x0, u0),  0              ]]
#
# so
#
#   c(x, u) = x~ * Q * x~ + u~ * R * u~ + O((x - x0)^3) + O((u - u0)^3)
#
# Solving the LQR problem for (A, B, Q, R), we get a controller K for which
# the optimal action is u0 + u, where u equals
#
#   u~ = K * x~
#
# with the last component dropped. In other words, u = u0 + [K00, K01] * [x - x0, 1]
#
# env: {step_torch: torch.tensor([state_dim]), torch.tensor([action_dim]) -> torch.tensor([state_dim]),
#       loss_torch: torch.tensor([state_dim]), torch.tensor([action_dim]) -> torch.tensor(1)}
# state: np.array([state_dim])
# action: np.array([action_dim])
def _get_lqr_env(env, state, action):
    # Step 1: Get information
    state_dim = len(state)
    action_dim = len(action)

    # Step 2: Convert to torch tensors
    state = torch.tensor(state, dtype=torch.float, requires_grad=True)
    action = torch.tensor(action, dtype=torch.float, requires_grad=True)

    # Step 3: Get the next state
    next_state = env.step_torch(state, action)

    # Step 4: Get the loss
    loss = env.loss_torch(state, action)

    # Step 5: Compute A

    # Step 5a: Compute A00 = fx(x, u)
    A00 = [torch.autograd.grad(next_state_component, state, create_graph=True)[0] for next_state_component in next_state]
    A00 = torch.cat(A00)
    A00 = A00.reshape([state_dim, state_dim])
    A00 = A00.detach().numpy()

    # Step 5b: Compute A01 = f(x, u)
    A01 = (next_state - state).detach().numpy()
    A01 = np.array([A01]).T

    # Step 5c: Compuate A10 = 0
    A10 = np.zeros([1, state_dim])

    # Step 5c: Set up A11 = 1
    A11 = np.ones([1, 1])

    # Step 5d: Concatenate matrices to get A
    A0 = np.concatenate([A00, A01], axis=1)
    A1 = np.concatenate([A10, A11], axis=1)
    A = np.concatenate([A0, A1], axis=0)

    # Step 6: Compute B

    # Step 6a: Compute B00 = fu(x, u)
    B00 = [torch.autograd.grad(next_state_component, action, create_graph=True)[0] for next_state_component in next_state]
    B00 = torch.cat(B00)
    B00 = B00.reshape([state_dim, action_dim])
    B00 = B00.detach().numpy()

    # Step 6b: Compute B01 = 0
    B01 = np.zeros([state_dim, 1])

    # Step 6c: Compute B10 = 0
    B10 = np.zeros([1, action_dim])

    # Step 6d: Compute B11 = 0
    B11 = np.zeros([1, 1])

    # Step 6e: Concatenate to get B
    B0 = np.concatenate([B00, B01], axis=1)
    B1 = np.concatenate([B10, B11], axis=1)
    B = np.concatenate([B0, B1], axis=0)

    # Step 7: Compute Q

    # Step 7a: Compute cx(x0, u0)
    cx = torch.autograd.grad(loss, state, create_graph=True)[0]

    # Step 7b: Compute Q00 = 0.5 * cxx(x0, u0)
    Q00 = [torch.autograd.grad(cx_component, state, create_graph=True)[0] for cx_component in cx]
    Q00 = torch.cat(Q00)
    Q00 = Q00.reshape([state_dim, state_dim])
    Q00 = Q00.detach().numpy()
    Q00 = 0.5 * Q00

    # Step 7c: Compute Q10 = 0.5 * cx(x0, u0)
    Q10 = 0.5 * cx
    Q10 = Q10.detach().numpy()
    Q10 = np.array([Q10])

    # Step 7d: Compute Q01 = 0.5 * cx(x0, u0)
    Q01 = Q10.T

    # Step 7e: Compute Q11 = c(x0, u0)
    Q11 = loss.detach().numpy()
    Q11 = np.array([[Q11]])

    # Step 7f: Concatenate matrices to get Q
    Q0 = np.concatenate([Q00, Q01], axis=1)
    Q1 = np.concatenate([Q10, Q11], axis=1)
    Q = np.concatenate([Q0, Q1], axis=0)

    # Step 8: Compute R

    # Step 8a: Compute cu(x0, u0)
    cu = torch.autograd.grad(loss, action, create_graph=True)[0]

    # Step 8b: Compute Q00 = 0.5 * cxx(x0, u0)
    R00 = [torch.autograd.grad(cu_component, action, create_graph=True)[0] for cu_component in cu]
    R00 = torch.cat(R00)
    R00 = R00.reshape([action_dim, action_dim])
    R00 = R00.detach().numpy()
    R00 = 0.5 * R00

    # Step 8c: Compute R10 = 0.5 * cu(x0, u0)
    R10 = 0.5 * cu
    R10 = R10.detach().numpy()
    R10 = np.array([R10])

    # Step 7d: Compute R01 = 0.5 * cu(x0, u0)
    R01 = R10.T

    # Step 7e: Compute R11 = 0
    R11 = np.zeros([action_dim, action_dim])

    # Step 7f: Concatenate matrices to get Q
    R0 = np.concatenate([R00, R01], axis=1)
    R1 = np.concatenate([R10, R11], axis=1)
    R = np.concatenate([R0, R1], axis=0)

    return LQREnv(A, B, Q, R)
