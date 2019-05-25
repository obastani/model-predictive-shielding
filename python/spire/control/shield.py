# A policy that ensures safety using online shielding.
#
# Note: In our implementation, the function is_stable
# can return membership in a subset H of a valid
# region of attraction (RoA) G_epsilon. Note that H
# may not itself be an RoA. However, our implementation
# slightly modifies the original shielding algorithm
# to allow use of such an H. In particular, once the
# system has reached H and started to use the LQR, it
# continues to use the LQR even if the system exits H.
# Since H \subseteq G_epsilon, we are guaranteed that
# the trajectory remains inside G_epsilon. Thus, the
# safety guarantee provided by our online shield still
# holds. We make this modification to allow subsets H
# for which membership is easier to compute than for
# the original RoA G_epsilon. If H = G_epsilon, then
# this implementation is equivalent to the original.
#
# env: GymEnv
# policy: Policy (the desired policy)
# rec_policy: Policy (the recovery policy)
# lqr_policy_gen: State -> Policy (function for generate a stabilizing policy for the given state)
# is_stable: State -> bool (whether a given state is in the region of attraction of the LQR policy)
# is_safe: State -> bool (whether a given state is considered safe)
# recovery_steps: int (number of steps to consider for recovery policy)
#
# Policy = {act: State -> Action}
# State = np.array([state_dim])
# Action = np.array([action_dim])
class ShieldPolicy:
    def __init__(self, env, policy, rec_policy, lqr_policy_gen, is_safe, is_stable, recovery_steps):
        self.env = env
        self.policy = policy
        self.rec_policy = rec_policy
        self.lqr_policy_gen = lqr_policy_gen
        self.is_safe = is_safe
        self.is_stable = is_stable
        self.recovery_steps = recovery_steps
        self.cur_lqr_policy = None

    # Get a safe action.
    #
    # state: State
    # return: Action
    def act(self, state):
        action, _ = self.act_shield(state)
        return action

    # Get a safe action, with the policy type.
    #
    # state: State
    # return: (Action, str)
    def act_shield(self, state):
        # Step 1: If the next state is recoverable, use the learned policy
        if self._is_recoverable(self.env.step(state, self.policy.act(state))):
            self.cur_lqr_policy = None
            return self.policy.act(state), 'learned'

        # Step 2: If already using LQR, use the LQR
        elif not self.cur_lqr_policy is None:
            return self.cur_lqr_policy.act(state), 'stable'

        # Step 3: If the state is stable, use the LQR
        elif self.is_stable(state):
            self.cur_lqr_policy = self.lqr_policy_gen(state)
            return self.cur_lqr_policy.act(state), 'stable'

        # Step 4: If the current state is recoverable, use the recovery policy
        elif self._is_recoverable(state):
            return self.rec_policy.act(state), 'recovery'

        # Step 5: Otherwise, use the LQR
        #
        # Note: We include this extra case as a default if the system
        # is not currently recoverable. Since our safety guarantee only
        # applies to recoverable states, the shield policy is anyway
        # not guaranteed to be safe from such as state. We find that
        # the LQR does a better job of keeping the system safe
        # compared to the recovery policy.
        else:
            self.cur_lqr_policy = self.lqr_policy_gen(state)
            return self.cur_lqr_policy.act(state), 'stable'

    # Check whether the state is recoverable,
    # i.e., the recovery controller can get it to the
    # region of attraction of the LQR controller.
    #
    # state: State
    # return: bool
    def _is_recoverable(self, state):
        for _ in range(self.recovery_steps):
            # Step 1: Check if the current state is stable
            if self.is_stable(state):
                return True

            # Step 2: Check if the current state is safe
            if not self.is_safe(state):
                return False

            # Step 3: Transition the system using the recovery policy
            state = self.env.step(state, self.rec_policy.act(state))

        # Step 4: Otherwise, the system is not recoverable
        return False
