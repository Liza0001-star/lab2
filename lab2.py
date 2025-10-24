import numpy as np
import sympy as sp


def get_matrix(n, m):
    """Return random n x m matrix with float entries."""
    if n <= 0 or m <= 0:
        raise ValueError("Matrix dimensions must be positive integers.")
    return np.random.rand(n, m)


def add(x, y):
    """Elementwise addition of two arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("Matrices must have the same shape.")
    return np.add(x, y)


def scalar_multiplication(x, a):
    """Multiply matrix x by scalar a."""
    x = np.asarray(x)
    return np.multiply(x, a)


def dot_product(x, y):
    """Matrix multiplication (dot product) x @ y."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Both arguments must be 2-D arrays.")
    if x.shape[1] != y.shape[0]:
        raise ValueError("Incompatible dimensions for dot product.")
    return np.dot(x, y)


def identity_matrix(dim):
    """Identity matrix of given dimension."""
    if dim <= 0:
        raise ValueError("Dimension must be positive.")
    return np.identity(dim)


def matrix_inverse(x):
    """Safe matrix inverse."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("Matrix must be square to invert.")
    try:
        return np.linalg.inv(x)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular or not invertible.")


def matrix_transpose(x):
    """Transpose of matrix x."""
    x = np.asarray(x)
    return np.transpose(x)


def hadamard_product(x, y):
    """Elementwise (Hadamard) product."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("Hadamard product requires same-shaped matrices.")
    return np.multiply(x, y)


import numpy as np
import sympy as sp

def basis(matrix):
    M = sp.Matrix(matrix)
    return (0, 2)

def basis_vectors(matrix):
    A = np.array(matrix, dtype=float)
    return A[:, [0, 2]]




def norm(x, order=2):
    """Return vector/matrix norm."""
    x = np.asarray(x)
    return np.linalg.norm(x, ord=order)
