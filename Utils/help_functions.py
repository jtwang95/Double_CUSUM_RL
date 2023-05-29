from asyncore import read
import numpy as np


def choose_time_points(T0, T1, epsilon=0.1, num=None, forced_time_points=None):
    T_range = T1 - T0 + 1
    if forced_time_points is not None:
        return [int(i) for i in forced_time_points]
    if num is not None:
        ts = np.linspace(T0 + int(T_range * epsilon),
                         T1 - int(T_range * epsilon) + 1,
                         num=num + 2)
        return sorted(list(set([int(i) for i in ts])))[1:-1]


if __name__ == "__main__":
    pass
