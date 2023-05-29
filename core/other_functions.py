import numpy as np


class array_normalizer():
    def __init__(self, data, axis) -> None:
        self.__mean = np.mean(data, axis=axis)
        self.__std = np.std(data, axis=axis)

    def normalize(self, x):
        return (x - self.__mean) / self.__std

    def denormalize(self, x, mean_fold=1, std_fold=1):
        return x * self.__std * std_fold + self.__mean * mean_fold