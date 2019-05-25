import random
import time
import numpy as np

# Compute a single rollout.
#
# env: Environment
# policy: Policy
# render: bool
# return: [(np.array([state_dim]), np.array([action_dim]), float, np.array([state_dim]))] ((state, action, loss, next_state) tuples)
def get_rollout(env, policy, render):
    # Step 1: Initialization
    state = env.reset()

    # Step 2: Compute rollout
    sarss = []
    for _ in range(env.max_steps()):
        # Step 2a: Action
        action = policy.act(state)

        # Step 2b: Render environment
        if render:
            env.render(state, action)

        # Step 2c: Transition environment
        next_state = env.step(state, action)

        # Step 2d: Get loss
        loss = env.loss(state, action)

        # Step 2e: Rollout (s, a, r)
        sarss.append((state, action, loss, next_state))

        # Step 2f: Update state
        state = next_state

    # Step 3: Render final state
    action = policy.act(state)
    if render:
        env.render(state, action)

    return sarss

# Compute a single rollout.
#
# env: Environment
# policy: Policy
# is_safe: State -> bool
# return: float, float, float, float
def get_rollout_test(env, policy, is_safe):
    # Step 1: Initialization
    t = time.time()
    state = env.reset()

    # Step 2: Compute rollout
    cum_loss = 0.0
    cum_safe = 0.0
    for _ in range(env.max_steps()):
        # Step 2a: Action
        action = policy.act(state)

        # Step 2b: Update values
        cum_loss += env.loss(state, action)
        cum_safe += is_safe(state)

        # Step 2c: Transition environment
        state = env.step(state, action)

    # Step 3: Computations
    final_loss = env.final_loss(state, policy.act(state))
    t = time.time() - t

    return cum_loss, final_loss, cum_safe / env.max_steps(), t

# Estimate the cumulative loss of the policy.
#
# env: Environment
# policy: Policy
# is_safe: State -> bool
# n_rollouts: int
# return: float, float (mean, std)
def test_policy(env, policy, is_safe, n_rollouts):
    values = []
    for i in range(n_rollouts):
        values.append(get_rollout_test(env, policy, is_safe))
    values = np.array(values)
    return np.mean(values, axis=0), np.std(values, axis=0)

# Compute a single rollout.
#
# env: Environment
# policy: Policy
# return: [str] (policy types)
def get_rollout_policy_type(env, shield_policy):
    # Step 1: Initialization
    state = env.reset()

    # Step 2: Compute rollout
    policy_types = []
    for _ in range(env.max_steps()):
        # Step 2a: Action
        action, policy_type = shield_policy.act_shield(state)

        # Step 2b: Policy type
        policy_types.append(policy_type)

        # Step 2c: Transition environment
        state = env.step(state, action)

    return policy_types

# Estimate the probability of safety of the policy.
#
# env: Environment
# policy: Policy
# n_rollouts: int
# return: [(float, float, float)] (count of number of uses of each policy type, in the order learned, stable, recovery)
def test_policy_type(env, shield_policy, n_rollouts):
    # Step 0: Parameters
    policy_type_names = ['learned', 'stable', 'recovery']

    # Step 1: Initialization
    policy_type_counts = [{policy_type: 0 for policy_type in policy_type_names} for _ in range(env.max_steps())]

    # Step 2: Get policy type counts
    for i in range(n_rollouts):
        policy_types = get_rollout_policy_type(env, shield_policy)
        for j in range(env.max_steps()):
            policy_type_counts[j][policy_types[j]] += 1

    # Step 3: Normalize policy type counts
    for j in range(env.max_steps()):
        for policy_type in policy_type_names:
            policy_type_counts[j][policy_type] /= n_rollouts

    # Step 4: Transform policy type counts
    transformed_counts = []
    for j in range(env.max_steps()):
        transformed_counts.append(tuple((policy_type_counts[j][policy_type] for policy_type in policy_type_names)))

    return transformed_counts
