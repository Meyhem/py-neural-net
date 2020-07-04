import numpy as np


def logistic_function(vec: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(vec))


class Layer:
    def __init__(self, rows: int, cols: int):
        self.weights: np.ndarray = np.random.rand(rows, cols)
        self.biases: np.ndarray = np.random.random(size=cols)

    def feed_forward(self, input_vector: np.ndarray) -> np.ndarray:
        return logistic_function(input_vector.dot(self.weights) + self.biases)

    def __str__(self):
        return "weights={}\nbiases={}".format(self.weights, self.biases)

    def __repr__(self):
        return str(self)


def forward_pass(nn: [Layer], vector: np.ndarray) -> np.ndarray:
    for layer in nn:
        vector = layer.feed_forward(vector)

    return vector


nn: [Layer] = [Layer(2, 4), Layer(4, 2)]

print(forward_pass(nn, np.array([1, 1])))


