import unittest
import numpy as np
import sympy as sp
from lab2 import (
    get_matrix, add, scalar_multiplication, dot_product,
    identity_matrix, matrix_inverse, matrix_transpose,
    hadamard_product, basis, basis_vectors, norm
)

class TestMatrixTools(unittest.TestCase):

    def test_get_matrix(self):
        A = get_matrix(2, 3)
        self.assertEqual(A.shape, (2, 3))
        self.assertTrue(np.all((A >= 0) & (A <= 1)))

    def test_add(self):
        x = np.ones((2, 2))
        y = np.eye(2)
        res = add(x, y)
        expected = np.array([[2., 1.], [1., 2.]])
        np.testing.assert_array_almost_equal(res, expected)

    def test_scalar_multiplication(self):
        x = np.array([[1, 2]])
        res = scalar_multiplication(x, 3)
        np.testing.assert_array_equal(res, np.array([[3, 6]]))

    def test_dot_product(self):
        x = np.array([[1, 2]])
        y = np.array([[3], [4]])
        res = dot_product(x, y)
        self.assertEqual(res, 11)

    def test_identity_matrix(self):
        I = identity_matrix(3)
        np.testing.assert_array_equal(I, np.eye(3))

    def test_matrix_inverse(self):
        A = np.array([[1, 2], [3, 4]])
        inv = matrix_inverse(A)
        np.testing.assert_array_almost_equal(inv, np.linalg.inv(A))

    def test_matrix_transpose(self):
        A = np.array([[1, 2, 3]])
        res = matrix_transpose(A)
        np.testing.assert_array_equal(res, np.array([[1], [2], [3]]))

    def test_hadamard_product(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 2]])
        res = hadamard_product(A, B)
        np.testing.assert_array_equal(res, np.array([[2, 0], [3, 8]]))

    def test_basis(self):
        A = np.array([[1, 2, 3],
                      [2, 4, 6],
                      [1, 0, 1]], dtype=float)
        pivots = basis(A)
        self.assertEqual(pivots, (0, 2))

    def test_basis_vectors(self):
        A = np.array([[1, 2, 3],
                      [2, 4, 6],
                      [1, 0, 1]], dtype=float)
        vecs = basis_vectors(A)
        expected = np.array([[1., 3.], [2., 6.], [1., 1.]])
        np.testing.assert_array_almost_equal(vecs, expected)

    def test_norm(self):
        x = np.array([3, 4])
        self.assertEqual(norm(x), 5.0)
        self.assertAlmostEqual(norm(x, 1), 7.0)

if __name__ == "__main__":
    unittest.main()
