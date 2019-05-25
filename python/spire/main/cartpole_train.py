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

    # bptt
    action_dim = 1
    n_bptt_iters = 10000
    learning_rate = 1e-3
    gamma = 0.99
    print_freq = 100

    # lqr parameters
    n_iters = 100
    lqr_state = np.array([0.0, 0.0, 0.0, 0.0])
    lqr_action = np.array([0.0])

    # shield parameters
    is_safe = is_cartpole_safe
    is_stable = is_cartpole_stable
    recovery_steps = 100

    # Step 1: Build parameters
    bptt_params = BPTTParams(action_dim, n_bptt_iters, learning_rate, gamma, print_freq)
    policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, fname)
    rec_policy_params = BPTTPolicyParams(input_size, hidden_size, output_size, dir, rec_fname)
    lqr_params = LQRParams(n_iters)

    # Step 2: Environment
    bptt_env = CartPoleMoveEnv()
    env = BPTTWrapperEnv(bptt_env)

    # Step 3: BPTT Policy
    policy = BPTTPolicy(policy_params)

    # Step 4: Load or learn policy
    if exists_file(dir, fname):
        policy.load()
    else:
        bptt(bptt_env, policy, bptt_params)
        policy.save()

    # Step 5: Cleanup
    env.close()

    # Step 6: Recovery environment
    bptt_rec_env = CartPoleRecoveryEnv(policy)
    rec_env = BPTTWrapperEnv(bptt_rec_env)

    # Step 7: Recovery BPTT policy
    rec_policy = BPTTPolicy(rec_policy_params)

    # Step 8: Load or learn policy
    if exists_file(dir, rec_fname):
        rec_policy.load()
    else:
        bptt(bptt_rec_env, rec_policy, bptt_params)
        rec_policy.save()

    # Step 9: Cleanup
    rec_env.close()

if __name__ == '__main__':
    main()
