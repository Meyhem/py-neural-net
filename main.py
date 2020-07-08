import numpy as np
import matplotlib.pyplot as p


def logistic_function(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


# simplified derivation according to https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
def dx_logistic_function(x: np.ndarray) -> np.ndarray:
    return x*(1 - x)


class Layer:
    def __init__(self, rows: int, cols: int):
        self.weights: np.ndarray = np.random.rand(rows, cols)
        self.biases: np.ndarray = np.random.random(size=cols) - 0.5

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        return logistic_function(input_vector.dot(self.weights) + self.biases)

    def __str__(self):
        return "weights={}\nbiases={}".format(self.weights, self.biases)

    def __repr__(self):
        return str(self)


def forward_pass(net: [Layer], value: np.ndarray) -> list:
    activations = []
    for layer in net:
        value = layer.forward(value)
        activations.append(value)

    return activations


def backward_pass(net: [Layer], y: np.ndarray, activations: [np.ndarray]) -> [np.ndarray]:
    deltas = []
    errors = []
    for i in reversed(range(len(net))):
        layer = net[i]

        if layer == net[-1]:
            errors.append(y - activations[-1])
            delta = errors[-1] * dx_logistic_function(activations[-1])
            deltas.append(delta)
        else:
            next_layer = net[i + 1]
            errors.append(np.dot(next_layer.weights, deltas[-1].T))
            delta = errors[-1].T * dx_logistic_function(activations[i])
            deltas.append(delta)

    return list(reversed(deltas)), list(reversed(errors))


def apply_deltas(net: [Layer], deltas: [np.ndarray], errors: [np.ndarray], lr: float):
    for i, l in enumerate(net):
        l.weights += deltas[i] * lr
        l.biases += errors[i] * lr


def train(net: [Layer], x: np.ndarray, y: np.ndarray):
    for xi, yi in zip(x, y):
        activations = forward_pass(net, xi)
        deltas, errors = backward_pass(net, yi, activations)
        apply_deltas(net, deltas, errors, 0.3)


X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.asarray([[0], [0], [0], [1]])
nn: [Layer] = [Layer(2, 3), Layer(3, 3), Layer(3, 1)]

for i in range(50000):
    train(nn, X, Y)

    if i % 1000 == 0:
        out = forward_pass(nn, X)[-1]
        print(Y - out)
        err = np.sum(np.square(Y - out))
        print(err)

print('predict', forward_pass(nn, np.asarray([1, 0]))[-1])
# print(forward_pass(nn, np.array([1,1])))

p.show()
