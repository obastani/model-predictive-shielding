import numpy as np

from ..util.rl import *
from ..util.log import *
from ..util.wrapper import *
from ..util.file import *
from ..env.cartpole import *
from ..control.bptt import *

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # Step 0: Parameters

    # bptt policy
    input_size = 4
    hidden_size = 200
    output_size = 1
    dir = '../data/bptt_cartpole'
    fname = 'policy.dat'
    rec_fname = 'rec_policy.dat'

    # shield parameters
    is_safe = is_cartpole_safe
    is_stable = is_cartpole_stable

    # testing
    n_test_rollouts = 100

    # Step 1: Build parameters
    policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, fname)

    # Step 2: BPTT Policy
    policy = BPTTPolicy(policy_params)
    policy.load()

    # Step 3: Environment
    bptt_shield_env = CartPoleMoveEnv()
    shield_env = BPTTWrapperEnv(bptt_shield_env)

    # Step 7: Test policy
    values, values_std = test_policy(shield_env, policy, is_safe, n_test_rollouts)
    log('Cumulative reward: {} {}'.format(values[0], values_std[0]), INFO)
    log('Cumulative final: {} {}'.format(values[1], values_std[1]), INFO)
    log('Safety probability: {} {}'.format(values[2], values_std[2]), INFO)
    log('Cumulative time: {} {}'.format(values[3], values_std[3]), INFO)
    get_rollout(shield_env, policy, True)

    # Step 8: Cleanup
    shield_env.close()

if __name__ == '__main__':
    main()
