import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def dx_relu(x: np.ndarray) -> np.ndarray:
    return 1. * (x > 0)


class Layer:
    def __init__(self, rows: int, cols: int):
        self.weights: np.ndarray = np.random.rand(rows, cols)
        self.biases: np.ndarray = np.random.random(size=cols) - 0.5

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        return relu(input_vector.dot(self.weights) + self.biases)

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
            delta = errors[-1] * dx_relu(activations[-1])
            deltas.append(delta)
        else:
            next_layer = net[i + 1]
            errors.append(np.dot(next_layer.weights, deltas[-1].T))
            delta = errors[-1].T * dx_relu(activations[i])
            deltas.append(delta)

    return list(reversed(deltas)), list(reversed(errors))


def apply_deltas(net: [Layer], deltas: [np.ndarray], errors: [np.ndarray], lr: float):
    for i, l in enumerate(net):
        l.weights += deltas[i] * lr
        l.biases += errors[i] * lr


def train(net: [Layer], x: np.ndarray, y: np.ndarray, lr: float):
    # run each input 100 times, MLP requires fewer epochs to learn that way
    for _ in range(100):
        for xi, yi in zip(x, y):
            activations = forward_pass(net, xi)
            deltas, errors = backward_pass(net, yi, activations)
            apply_deltas(net, deltas, errors, lr)


X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.asarray([[0], [0], [0], [1]])
nn: [Layer] = [Layer(2, 3), Layer(3, 3), Layer(3, 1)]

epochs = 0
while True:
    train(nn, X, Y, 0.05)
    out = forward_pass(nn, X)[-1]
    err = np.sum(np.square(Y - out))

    if err < 0.01:
        print('Trained in {} epochs. SSE={}'.format(epochs, err))
        break
    epochs += 1

print('predict 0 & 0', forward_pass(nn, np.asarray([0, 0]))[-1])
print('predict 1 & 0', forward_pass(nn, np.asarray([1, 0]))[-1])
print('predict 0 & 1', forward_pass(nn, np.asarray([0, 1]))[-1])
print('predict 1 & 1', forward_pass(nn, np.asarray([1, 1]))[-1])
