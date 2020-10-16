import sys
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


np.random.seed(42)
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

