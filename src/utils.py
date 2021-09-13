"""
Vector-related computing for debugging and testing.
"""

import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
import time

import tensorflow as tf


def vec_length(vec):
    return LA.norm(vec)

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

class IndexScore:
    """
    The score of a tail when h and r is given. Or the distance of a vector to the target in kNN sampling.
    It's used in the ranking task to facilitate comparison and sorting.
    Print score as 3 digit precision float.
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return "(index: %d, w:%.3f)" % (self.index, self.score)
        return "(%d, %.3f)" % (self.index, self.score)

    def __str__(self):
        return "(index: %d, score:%.3f)" % (self.index, self.score)