import numpy as np
import gym
import time

from rl import *
from dt import *
from log import *
from cartpole import *

dirname = '../data'
fname = 'dt_policy.pk'

def evaluate_dt():
    # Step 0: Parameters
    n_test_rollouts = 100

    # Step 1: Data structures
    policy = load_dt_policy(dirname, fname)

    # Step 2: Environment
    env = gym.make('CartPole-v0')

    # Step 3: Evaluate policy
    rew = test_policy(env, policy, lambda obs: obs, n_test_rollouts)
    log('Reward: {}'.format(rew), INFO)

def solve(t_max):
    # Step 1: Data structures
    policy = load_dt_policy(dirname, fname).tree
    policy_func = lambda z: make_policy_func(policy, z)

    # Step 2: Verification
    t = time.time()
    solve_policy(policy_func, t_max)
    log(t_max, INFO)
    log(time.time() - t, INFO)
    log('', INFO)

if __name__ == '__main__':
    t_max = 10
    solve(t_max)
