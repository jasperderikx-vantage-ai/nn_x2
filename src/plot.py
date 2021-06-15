import matplotlib.pyplot as plt
import numpy as np


def plot(functions, file_name):
    """Plot functions on interval [0,1)."""
    sample = np.arange(0.0, 1.0, 0.01)
    fig, ax = plt.subplots()

    for f in functions:
        y = [f(x) for x in sample]
        ax.plot(sample, y)

    fig.savefig(file_name)
