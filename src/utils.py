
import numpy as np

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
