import keras
import tensorflow as tf
import numpy as np
import os
from src.plot import plot

np.random.seed(1)
tf.random.set_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_dataset(function, data_size):
    """Generate random sample of size 'data_size' on interval [0,1) with its image according to 'function'."""
    xs = np.random.rand(data_size)
    ys = function(xs)
    return np.array([[x] for x in xs]), np.array([[y] for y in ys])


def fit_nn(function, data_size=1000):
    """Fit a neural network to the given function."""
    model = keras.Sequential([
        keras.layers.Dense(1, activation='linear', input_shape=(1,)),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dense(1, activation='linear'),
    ])

    model.compile(optimizer='Adam', loss=tf.losses.MSE)

    data, labels = create_dataset(function=function, data_size=data_size)
    model.fit(x=data, y=labels, epochs=100, validation_split=0.1)
    return model


def construct_nn(function, n_bars=10, precision=100):
    """Construct a neural network that mimics the given function."""
    model = keras.Sequential([
        keras.layers.Dense(1, activation='linear', input_shape=(1,)),
        keras.layers.Dense(n_bars, activation='sigmoid'),
        keras.layers.Dense(1, activation='linear'),
    ])

    bar_width = 1 / n_bars
    offset_x = bar_width / 2
    xs = [offset_x + i * bar_width for i in range(n_bars)]
    y_deltas = [function(xs[0])] + [function(xs[i + 1]) - function(xs[i]) for i in range(len(xs) - 1)]

    model.layers[0].set_weights([np.array([[1]]), np.array([0])])
    model.layers[1].set_weights([np.array([[precision] * n_bars]),
                                 np.array([-precision * i / n_bars for i in range(n_bars)])])
    model.layers[2].set_weights([np.array([[y] for y in y_deltas]), np.array([0])])

    model.compile(optimizer='Adam', loss=tf.losses.MSE)
    return model


def functify(nn):
    """Refactor a neural network into a function on the Euclidian plane."""

    def f(x):
        y = nn(np.array([[x]]))
        return float(y)

    return f


square_function = lambda x: x ** 2
nn_fitted = fit_nn(square_function)
nn_constructed = construct_nn(square_function)

plot([functify(nn_fitted), square_function], "../data/nn_fitted.png")
plot([functify(nn_constructed), square_function], "../data/nn_constructed.png")
