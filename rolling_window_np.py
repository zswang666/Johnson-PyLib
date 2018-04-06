import numpy as np
from functools import reduce

from pprint import pprint

def test():
    mat = np.arange(16).reshape((4,4))
    win = [2,2]
    strd = [2, 2]
    mat2 = rolling_window(mat, win, strd)

    pprint(mat)
    pprint(mat2)

    mat = np.arange(9)
    win = [3]
    strd = [2]
    mat2 = rolling_window(mat, win, strd)
    pprint(mat)
    pprint(mat2)


def rolling_window(a, window, strides):
    """ Perform rolling/sliding window over a multi-dimensional array
            Args:
                a (np.ndarray): array to perform rolling window to
                window (tuple or list): window array to be rolled
                strides (tuple or list): strides for rolling window
            Returns:
                patches (np.ndarray): all sliding window on `a` defined by `window` and `strides` 
            Notes:
                1. For performance, this function do not perform sanity check. Please use it carefully.
            Examples:
                >> mat = np.arange(16).reshape((4,4))
                >> mat2 = rolling_window(mat, [2,2], [2,2])
                >> print(mat)
                array([[ 0,  1,  2,  3],
                       [ 4,  5,  6,  7],
                       [ 8,  9, 10, 11],
                       [12, 13, 14, 15]])
                >> mat2.shape
                (2, 2, 2, 2)
                >> print(mat2[0,0])
                array([[0, 1],
                       [4, 5]])
                >> print(mat2[0,1])
                array([[2, 3],
                       [6, 7]])
                >> print(mat2[1,0])
                array([[8, 9],
                       [12, 13]])
                >> print(mat2[1,1])
                array([[10, 11],
                       [14, 15]])
    """
    assert len(a.shape)==len(window)==len(strides), "\'a\', \'window\', \'strides\' dimension mismatch"
    shape_fn = lambda i,w,s: (a.shape[i]-w)//s + 1
    shape = [shape_fn(i,w,s) for i,(w,s) in enumerate(zip(window, strides))] + list(window)
    def acc_shape(i):
        if i+1>=len(a.shape):
            return 1
        else:
            return reduce(lambda x,y:x*y, a.shape[i+1:])
    _strides = [acc_shape(i)*s*a.itemsize for i,s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)

if __name__=="__main__":
    test()