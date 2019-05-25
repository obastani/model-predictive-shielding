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
    recovery_steps = 100 # this is the parameter T in our algorithm

    # testing
    n_test_rollouts = 100
    is_policy_counts = True

    # Step 1: Build parameters
    policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, fname)
    rec_policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, rec_fname)

    # Step 2: Environment
    bptt_env = BicycleEnv()
    env = BPTTWrapperEnv(bptt_env)

    # Step 3: BPTT Policy
    policy = BPTTPolicy(policy_params)
    policy.load()

    # Step 4: Recovery environment
    bptt_rec_env = BicycleRecoveryEnv(policy)
    rec_env = BPTTWrapperEnv(bptt_rec_env)

    # Step 5: Recovery BPTT policy
    rec_policy = BPTTPolicy(rec_policy_params)
    rec_policy.load()

    # Step 6: Safe policy
    safe_policy = BicycleSafePolicy()
    safe_policy_gen = lambda state: safe_policy

    # Step 7: Shield policy
    shield_policy = ShieldPolicy(env, policy, rec_policy, safe_policy_gen, is_safe, is_stable, recovery_steps)

    # Step 8: Test policy
    values, values_std = test_policy(env, shield_policy, is_safe, n_test_rollouts)
    log('Cumulative reward: {} {}'.format(values[0], values_std[0]), INFO)
    log('Cumulative final: {} {}'.format(values[1], values_std[1]), INFO)
    log('Safety probability: {} {}'.format(values[2], values_std[2]), INFO)
    log('Cumulative time: {} {}'.format(values[3], values_std[3]), INFO)
    if is_policy_counts:
        policy_type_counts = test_policy_type(env, shield_policy, n_test_rollouts)
        for policy_type_count in policy_type_counts:
            log('{} {} {}'.format(policy_type_count[0], policy_type_count[1], policy_type_count[2]), INFO)
    get_rollout(env, shield_policy, True)

    # Step 9: Cleanup
    env.close()

if __name__ == '__main__':
    main()
