import numpy as np

from ..util.rl import *
from ..util.log import *
from ..util.wrapper import *
from ..util.file import *
from ..env.bicycle import *
from ..control.bptt import *
from ..control.shield import *

def main():
    # Step 0: Parameters

    # bptt policy
    input_size = 7
    hidden_size = 200
    output_size = 2
    dir = '../data/bptt_bicycle'
    fname = 'policy.dat'
    rec_fname = 'rec_policy.dat'

    # bptt
    action_dim = 1
    n_bptt_iters = 10000
    learning_rate = 1e-5
    gamma = 0.99
    print_freq = 100

    # obstacle parameters
    obstacle_radius = 0.05
    x_obstacle_0 = 0.4
    x_obstacle_1 = 0.7

    # shield parameters
    is_safe = lambda state: is_bicycle_safe(state, x_obstacle_0, x_obstacle_1, obstacle_radius)
    is_stable = lambda state: is_bicycle_stable(state, x_obstacle_0, x_obstacle_1, obstacle_radius)
    recovery_steps = 100

    # Step 1: Build parameters
    bptt_params = BPTTParams(action_dim, n_bptt_iters, learning_rate, gamma, print_freq)
    policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, fname)
    rec_policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, rec_fname)

    # Step 2: Environment
    bptt_env = BicycleEnv()
    env = BPTTWrapperEnv(bptt_env)

    # Step 3: BPTT Policy
    policy = BPTTPolicy(policy_params)

    # Step 4: Load or learn policy
    if exists_file(dir, fname):
        policy.load()
    else:
        bptt(bptt_env, policy, bptt_params)
        policy.save()

    # Step 5: Recovery environment
    bptt_rec_env = BicycleRecoveryEnv(policy)
    rec_env = BPTTWrapperEnv(bptt_rec_env)

    # Step 6: Recovery BPTT policy
    rec_policy = BPTTPolicy(rec_policy_params)

    # Step 7: Load or learn policy
    if exists_file(dir, rec_fname):
        rec_policy.load()
    else:
        bptt(bptt_rec_env, rec_policy, bptt_params)
        rec_policy.save()

    # Step 8: Cleanup
    rec_env.close()

if __name__ == '__main__':
    main()
