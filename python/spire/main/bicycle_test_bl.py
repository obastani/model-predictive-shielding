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

    # obstacle parameters
    obstacle_radius = 0.05
    #obstacle_radius = 0.2
    x_obstacle_0 = 0.4
    x_obstacle_1 = 0.7

    # shield parameters
    is_safe = lambda state: is_bicycle_safe(state, x_obstacle_0, x_obstacle_1, obstacle_radius)
    is_stable = lambda state: is_bicycle_stable(state, x_obstacle_0, x_obstacle_1, obstacle_radius)

    # testing
    n_test_rollouts = 100

    # Step 1: Build parameters
    policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, fname)

    # Step 2: Environment
    bptt_env = BicycleEnv()
    env = BPTTWrapperEnv(bptt_env)

    # Step 3: BPTT Policy
    policy = BPTTPolicy(policy_params)
    policy.load()

    # Step 4: Test policy
    values, values_std = test_policy(env, policy, is_safe, n_test_rollouts)
    log('Cumulative reward: {} {}'.format(values[0], values_std[0]), INFO)
    log('Cumulative final: {} {}'.format(values[1], values_std[1]), INFO)
    log('Safety probability: {} {}'.format(values[2], values_std[2]), INFO)
    log('Cumulative time: {} {}'.format(values[3], values_std[3]), INFO)
    get_rollout(env, policy, True)

    # Step 9: Cleanup
    env.close()

if __name__ == '__main__':
    main()
