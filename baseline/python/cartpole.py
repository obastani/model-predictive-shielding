import numpy as np
from z3 import *

from log import *

A = np.array([[ 0.0, 1.0,  0.0,    0.0 ],
              [ 0.0, 0.0, -7.1707, 0.0 ],
              [ 0.0, 0.0,  0.0,    1.0 ],
              [ 0.0, 0.0,  7.8878, 0.0 ]])
    
B = np.array([ 0.0, 1.5743, 0.0, -0.7317 ])

force = 10.0
dt = 0.02

def f(z, a):
    u = -force if a == 0 else force
    y = z + (np.matmul(A, z) + u * B) * dt
    return y

def f_symb(z, a):
    u = If(a == 0, -force, force)
    y = z.copy()
    for i in range(4):
        for j in range(4):
            y[i] += A[i,j] * z[j] * dt
        y[i] += u * B[i] * dt
    return y

def get_vars(t):
    return [Real('x' + str(t)), Real('v' + str(t)), Real('t' + str(t)), Real('w' + str(t))]

def get_init(z):
    return And(z[0] >= -0.05, z[0] <= 0.05,
               z[1] >= -0.05, z[1] <= 0.05,
               z[2] >= -0.05, z[2] <= 0.05,
               z[3] >= -0.05, z[3] <= 0.05)

def get_safety(z):
    return And(z[2] >= -0.1, z[2] <= 0.1)

def solve_policy(policy, t_max):

    # Step 0: Solver
    s = Solver()

    # Step 1: Get initial variables
    z = get_vars(0)

    # Step 2: Build initial constraints
    s.add(get_init(z))

    # Step 3: Run dynamics
    safety = []
    for t in range(t_max):
        # dynamics
        z = f_symb(z, policy(z))

        # safety
        safety.append(get_safety(z))

        # logging
        zt = get_vars(t+1)
        s.add([z[i] == zt[i] for i in range(4)])

    # Step 4: Safety constraint
    s.append(Not(And(safety)))

    # Step 5: Run solver
    r = s.check()
    log('Result: {}'.format(r), INFO)
    if r == sat:
        for t in range(t_max+1):
            log('State {}: {}'.format(t, [_convert(s.model()[var]) for var in get_vars(t)]), INFO)
    
def _convert(val):
    if val.__class__ == RatNumRef:
        return float(val.numerator_as_long())/float(val.denominator_as_long())
    elif val.__class__ == IntNumRef:
        return val.as_long()
    else:
        raise Exception()

def make_policy_func(policy, z):
    return _make_policy_func_helper(policy, z, 0)

def _make_policy_func_helper(policy, z, nid):
    # Step 0: Base case (leaf node)
    if policy.tree_.children_left[nid] == policy.tree_.children_right[nid]:
        return int(np.argmax(policy.tree_.value[nid]))
    
    # Step 1: Feature
    s = z[policy.tree_.feature[nid]]

    # Step 2: Threshold
    t = policy.tree_.threshold[nid]

    # Step 3: Recursive calls
    v_true = _make_policy_func_helper(policy, z, policy.tree_.children_left[nid])
    v_false = _make_policy_func_helper(policy, z, policy.tree_.children_right[nid])

    # Step 4: Construct if statement
    return If(s <= t, v_true, v_false)

