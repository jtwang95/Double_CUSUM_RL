import numpy as np
from core.neural_nets import *
import logging

mylogger = logging.getLogger("testSA2")


class random_sin_funcion():
    def __init__(self, s_dim=1):
        self.theta1 = np.random.normal(loc=0, scale=1, size=[s_dim + 1, 1])
        self.theta0 = np.random.normal(loc=0, scale=1, size=1)

    def call(self, x):
        return (np.sin(np.matmul(x, self.theta1) + self.theta0))


class random_h():
    def __init__(self, s_dim=1, htype="nn") -> None:
        self.s_dim = s_dim
        if htype == "nn":
            # mylogger.debug("Using nn as random h function")
            self.model = anotherSmallNet(self.s_dim + 1, 1, hidden_dims=[16])
        elif htype == "sincos":
            # mylogger.debug("Using sincos as random h function")
            self.model = random_sin_funcion(self.s_dim)
        elif htype == "reward":
            # mylogger.debug("Using reward as h function")

            class reward_h_fn():
                def call(self, x):
                    return x[:, -1]

            self.model = reward_h_fn()
        elif htype == "nn_state":
            # mylogger.debug("Using nn as random h function with only state")
            model = anotherSmallNet(self.s_dim, 1, hidden_dims=[16])

            class state_h_fn():
                def __init__(self, m) -> None:
                    self.m = m

                def call(self, x):
                    return self.m.call(x[:, :-1])

            self.model = state_h_fn(model)

    def call(self, sp, r):
        sp = np.array(sp).reshape([-1, self.s_dim])
        r = np.array(r).reshape([-1, 1])
        # x = (np.concatenate([sp, r], axis=1) -
        #      self.normalize_mean[np.newaxis, :]
        #      ) / self.normalize_sd[np.newaxis, :]
        x = np.concatenate([sp, r], axis=1)
        return self.model.call(x)


if __name__ == "__main__":
    s = np.random.normal(0, 1, [3, 10])
    r = np.random.normal(0, 1, [3, 1])
    h0 = random_h(htype="nn", s_dim=10, S=s, R=r)
    print(h0.call(s, r))
