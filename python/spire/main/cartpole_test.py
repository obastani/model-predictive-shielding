import numpy as np

from ..util.rl import *
from ..util.log import *
from ..util.wrapper import *
from ..util.file import *
from ..env.cartpole import *
from ..control.bptt import *
from ..control.lqr import *
from ..control.shield import *

def main():
    # Step 0: Parameters

    # bptt policy
    input_size = 4
    hidden_size = 200
    output_size = 1
    dir = '../data/bptt_cartpole'
    fname = 'policy.dat'
    rec_fname = 'rec_policy.dat'

    # lqr parameters
    n_iters = 100
    lqr_state = np.array([0.0, 0.0, 0.0, 0.0])
    lqr_action = np.array([0.0])

    # shield parameters
    is_safe = is_cartpole_safe
    is_stable = is_cartpole_stable
    recovery_steps = 100 # this is the parameter T in our algorithm

    # testing
    n_test_rollouts = 100
    is_policy_counts = True

    # Step 1: Build parameters
    policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, fname)
    rec_policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, rec_fname)
    lqr_params = LQRParams(n_iters)

    # Step 2: BPTT Policy
    policy = BPTTPolicy(policy_params)
    policy.load()

    # Step 3: Recovery BPTT policy
    rec_policy = BPTTPolicy(rec_policy_params)
    rec_policy.load()

    # Step 4: LQR generator
    lqr_policy_gen = lambda state: lqr_nonlinear(CartPoleStableEnv(np.array([state[0], 0.0, 0.0, 0.0])), np.array([state[0], 0.0, 0.0, 0.0]), lqr_action, lqr_params)

    # Step 5: Environment
    bptt_shield_env = CartPoleMoveEnv()
    shield_env = BPTTWrapperEnv(bptt_shield_env)

    # Step 6: Shield policy
    shield_policy = ShieldPolicy(shield_env, policy, rec_policy, lqr_policy_gen, is_safe, is_stable, recovery_steps)

    # Step 7: Test policy
    values, values_std = test_policy(shield_env, shield_policy, is_safe, n_test_rollouts)
    log('Cumulative reward: {} {}'.format(values[0], values_std[0]), INFO)
    log('Cumulative final: {} {}'.format(values[1], values_std[1]), INFO)
    log('Safety probability: {} {}'.format(values[2], values_std[2]), INFO)
    log('Cumulative time: {} {}'.format(values[3], values_std[3]), INFO)
    if is_policy_counts:
        policy_type_counts = test_policy_type(shield_env, shield_policy, n_test_rollouts)
        for policy_type_count in policy_type_counts:
            log('{} {} {}'.format(policy_type_count[0], policy_type_count[1], policy_type_count[2]), INFO)
    get_rollout(shield_env, shield_policy, True)

    # Step 8: Cleanup
    shield_env.close()

if __name__ == '__main__':
    main()
