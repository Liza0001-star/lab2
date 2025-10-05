import numpy as np

def get_matrix(n, m):
    if n <= 0 or m <= 0:
        raise ValueError("Matrix dimensions must be positive integers.")
    return np.random.rand(n, m)

def add(x, y):
    if x.shape != y.shape:
        raise ValueError("Matrices must have the same shape.")
    return np.add(x, y)

def scalar_multiplication(x, a):
    return np.multiply(x, a)

def dot_product(x, y):
    if x.shape[1] != y.shape[0]:
        raise ValueError("Incompatible dimensions for dot product.")
    return np.dot(x, y)

def identity_matrix(dim):
    return np.identity(dim)

def matrix_inverse(x):
    if np.linalg.det(x) == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return np.linalg.inv(x)

def matrix_transpose(x):
    return np.transpose(x)

def hadamard_product(x, y):
    if x.shape != y.shape:
        raise ValueError("Hadamard product requires same-shaped matrices.")
    return np.multiply(x, y)

def basis(x):
    _, inds = np.linalg.qr(x, mode='reduced')
    return tuple(inds)

def norm(x, order):
    return np.linalg.norm(x, ord=order)
