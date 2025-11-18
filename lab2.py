import numpy as np
from typing import Tuple


def get_matrix(n: int, m: int) -> np.ndarray:
    """
    Generate a random n x m matrix with float entries in [0, 1).

    """
    if not (isinstance(n, int) and isinstance(m, int)):
        raise TypeError("Dimensions must be integers.")
    if n <= 0 or m <= 0:
        raise ValueError("Matrix dimensions must be positive integers.")
    return np.random.rand(n, m)


def add(x, y) -> np.ndarray:
    """
    Elementwise addition of two arrays.

    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError("Matrices must have the same shape.")
    return x_arr + y_arr


def scalar_multiplication(x, a) -> np.ndarray:
    """
    Multiply matrix `x` by scalar `a`.

    """
    x_arr = np.asarray(x)
    try:
        return x_arr * a
    except Exception as e:
        raise TypeError("Scalar multiplication failed.") from e


def dot_product(x, y):
    """
    Matrix multiplication (dot product) x @ y.

    If the result is a single element (1x1), returns a Python scalar.

    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("Both arguments must be 2-D arrays.")
    if x_arr.shape[1] != y_arr.shape[0]:
        raise ValueError("Incompatible dimensions for dot product.")
    res = x_arr @ y_arr
    # return scalar if 1x1
    if np.isscalar(res) or res.shape == ():
        return res.item() if hasattr(res, "item") else res
    if isinstance(res, np.ndarray) and res.size == 1:
        return res.item()
    return res


def identity_matrix(dim: int) -> np.ndarray:
    """
    Identity matrix of given dimension.

    """
    if not isinstance(dim, int):
        raise TypeError("Dimension must be an integer.")
    if dim <= 0:
        raise ValueError("Dimension must be positive.")
    return np.eye(dim)


def matrix_inverse(x) -> np.ndarray:
    """
    Safe matrix inverse.

    """
    x_arr = np.asarray(x)
    if x_arr.ndim != 2 or x_arr.shape[0] != x_arr.shape[1]:
        raise ValueError("Matrix must be square to invert.")
    try:
        return np.linalg.inv(x_arr)
    except np.linalg.LinAlgError as e:
        raise ValueError("Matrix is singular or not invertible.") from e


def matrix_transpose(x) -> np.ndarray:
    """
    Transpose of matrix x.

    """
    return np.asarray(x).T


def hadamard_product(x, y) -> np.ndarray:
    """
    Elementwise (Hadamard) product.

    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError("Hadamard product requires same-shaped matrices.")
    return x_arr * y_arr


def basis(matrix) -> Tuple[int, ...]:
    """
    Return tuple of indices of linearly independent columns (basis for the
    column space), determined by incremental rank checking.
   
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("Input must be a 2-D matrix.")
    n_rows, n_cols = A.shape
    independent_cols = []
    if n_cols == 0:
        return tuple()

    
    current_basis = np.empty((n_rows, 0), dtype=float)
    current_rank = 0

    for j in range(n_cols):
        col = A[:, j].reshape(-1, 1)
        candidate = np.hstack((current_basis, col)) if current_basis.size else col
    
        cand_rank = np.linalg.matrix_rank(candidate)
        if cand_rank > current_rank:
            independent_cols.append(j)
            current_basis = candidate
            current_rank = cand_rank

    return tuple(independent_cols)


def basis_vectors(matrix) -> np.ndarray:
    """
    Return basis vectors as columns of the original matrix corresponding to
    the pivot column indices returned by `basis()`.
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("Input must be a 2-D matrix.")
    pivot_cols = basis(A)
    if len(pivot_cols) == 0:
        
        return np.empty((A.shape[0], 0), dtype=float)
    return A[:, list(pivot_cols)]


def norm(x, order: int = 2):
    """
    Compute vector or matrix norm.
    """
    return np.linalg.norm(np.asarray(x), ord=order)

if __name__ == "__main__":
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [1, 1, 1]], dtype=float)

    print("Matrix A:")
    print(A)

    print("Pivot columns:", basis(A))
    print("Basis vectors:")
    print(basis_vectors(A))

    print("Dot product:")
    print(dot_product([[1,2]], [[3],[4]]))  # = 11

    print("Inverse of I:")
    print(matrix_inverse(identity_matrix(3)))

