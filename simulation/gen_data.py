import numpy as np
import logging
from scipy.special import expit

mylogger = logging.getLogger("testSA")


class multiGaussionSys():
    def __init__(self, s_dim=1):
        self.s_dim = s_dim
        self.transition_fn = None
        self.reward_fn = None

    def transition_homo(self, state, action, t):
        n = action.shape[0]
        return 0.5 * state + np.random.normal(self.mean, self.cov,
                                              [n, self.s_dim])

    def reward_homo(self, state, action, t):
        n = action.shape[0]
        a = np.ones([1, self.s_dim]) * np.array(range(self.s_dim)).reshape(
            [1, -1])
        A = np.matmul((2.0 * action - 1.0).reshape([-1, 1]), a)
        b = state[:, 0] + state[:, -1] + state[:, 0] * state[:, -1] + np.abs(
            state[:, 0])
        return b + np.random.normal(self.mean, self.cov, n)

    def transition_pwc2ada_reward(self, state, action, t):
        n = action.shape[0]
        a = np.ones([1, self.s_dim])
        A = 0.5 * np.matmul(action.reshape([-1, 1]), a)
        return np.multiply(A, state) + np.random.normal(
            self.mean, self.cov, [n, self.s_dim])

    def reward_pwc2ada_reward(self, state, action, t):
        n = action.shape[0]
        beta = 1 * self.s_dim  #* self.s_dim  #* self.s_dim
        v = np.array(t < self.chg_pt, dtype=float)
        b1 = -1.5 * beta * np.mean(state, axis=1, keepdims=True)
        b2 = 1 * beta * np.mean(state, axis=1, keepdims=True)
        return v * b1 + (1 -
                         v) * b2  #+ np.random.normal(self.mean, 0.1, [n, 1])

    def transition_pwc2ada_state(self, state, action, t):
        n = action.shape[0]
        a = np.ones([1, self.s_dim])
        v = np.array(t < self.chg_pt, dtype=float)
        eta = 0.5 * np.matmul(action, a)
        s1 = -1 * np.multiply(eta, state)
        s2 = 1 * (np.log10(self.s_dim) + 0.5) * np.multiply(eta, state)
        return v * s1 + (1 - v) * s2 + np.random.normal(
            self.mean, self.cov, [n, self.s_dim])

    def reward_pwc2ada_state(self, state, action, t):
        n = action.shape[0]
        beta = 1  #* self.s_dim  #* self.s_dim
        s1 = beta * np.mean(state, axis=1, keepdims=True)
        b1 = 0.25 * np.multiply(action, np.power(s1, 2)) + 4 * s1
        return b1  #+ np.random.normal(self.mean, 0.1, [n, 1])

    def transition_hmoada(self, state, action, t):
        n = action.shape[0]
        a = np.ones([1, self.s_dim])
        A = 0.5 * np.matmul((2.0 * action - 1.0).reshape([-1, 1]), a)
        return np.multiply(A, state) + np.random.normal(
            self.mean, self.cov, [n, self.s_dim])

    def reward_hmoada(self, state, action, t):
        n = action.shape[0]
        beta = 1 * self.s_dim  #* self.s_dim  #* self.s_dim
        b1 = -0.5 * beta * np.mean(state, axis=1, keepdims=True)
        return b1

    def simulate(self,
                 mean0=0,
                 cov0=1,
                 policy=None,
                 T=100,
                 N=2,
                 type="hmo",
                 mean=0,
                 cov=1,
                 burnin=1000,
                 change_pt=None):
        '''
        Simulate data
        :param mean0: mean of initial state
        :param cov0: cov of initial state
        :param policy: policy function S |-> A
        :param T: number of time points
        :param N: number of trajectories
        :param type: simulation type
        :param mean: mean of state transition noise
        :param cov: cov of state transition noise
        '''
        self.n = N
        self.mean = mean
        self.cov = cov
        self.type = type
        if type == "hmo":
            self.chg_pt = np.nan
            self.transition_fn = self.transition_homo
            self.reward_fn = self.reward_homo

        if type == "pwc2ada_reward":
            self.chg_pt = change_pt if change_pt else int(T / 2)
            mylogger.debug("The change point is {}.".format(self.chg_pt))
            self.transition_fn = self.transition_pwc2ada_reward
            self.reward_fn = self.reward_pwc2ada_reward

        if type == "pwc2ada_state":
            self.chg_pt = change_pt if change_pt else int(T / 2)
            mylogger.debug("The change point is {}.".format(self.chg_pt))
            self.transition_fn = self.transition_pwc2ada_state
            self.reward_fn = self.reward_pwc2ada_state

        if type == "hmoada":
            self.chg_pt = np.nan
            self.transition_fn = self.transition_hmoada
            self.reward_fn = self.reward_hmoada

        # traj
        S = np.zeros([N, T + 1, self.s_dim])
        A = np.zeros([N, T], dtype=np.int8)
        R = np.zeros([N, T])

        # burnin
        # init state
        state = np.random.normal(mean0, cov0, [self.n, self.s_dim])
        for t in range(burnin):
            action = np.random.binomial(1, 0.5, self.n).reshape([-1, 1])
            # get immediate reward from the env
            reward = self.reward_fn(state=state,
                                    action=action,
                                    t=np.repeat([[0]], axis=0, repeats=self.n))
            # transit to another state given current action, state and immediate reward
            state = self.transition_fn(state=state,
                                       action=action,
                                       t=np.repeat([[0]],
                                                   axis=0,
                                                   repeats=self.n))

        # simulate
        S[:, 0, :] = state
        for t in range(T):
            if policy == None:
                # random policy (choose an action)
                if t < int(0.7 * T):
                    action = np.random.binomial(1, 0.5,
                                                self.n).reshape([-1, 1])
                else:
                    action = np.random.binomial(1, 0.8,
                                                self.n).reshape([-1, 1])
            else:
                action = policy(state, t)
            # get immediate reward from the env
            reward = self.reward_fn(state=state,
                                    action=action,
                                    t=np.repeat([[t]], axis=0, repeats=self.n))
            # transit to another state given current action, state and immediate reward
            state = self.transition_fn(state=state,
                                       action=action,
                                       t=np.repeat([[t]],
                                                   axis=0,
                                                   repeats=self.n))
            S[:, t + 1, :] = state
            A[:, t] = action.flatten()
            R[:, t] = reward.flatten()
        return S, A, R


if __name__ == "__main__":
    pass
