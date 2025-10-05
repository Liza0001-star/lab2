import numpy as np
from lab2 import (  
    get_matrix,
    add,
    scalar_multiplication,
    dot_product,
    identity_matrix,
    matrix_inverse,
    matrix_transpose,
    hadamard_product,
    basis,
    norm
)

def main():
    # 1. Тест створення матриці
    A = get_matrix(3, 3)
    B = get_matrix(3, 3)
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    # 2. Додавання матриць
    print("\nA + B =\n", add(A, B))

    # 3. Множення на скаляр
    print("\n2 * A =\n", scalar_multiplication(A, 2))

    # 4. Матричний добуток
    print("\nA @ B =\n", dot_product(A, B))

    # 5. Одинична матриця
    I = identity_matrix(3)
    print("\nIdentity matrix (3x3):\n", I)

    # 6. Обернена матриця
    C = np.array([[2, 1], [7, 4]], dtype=float)
    print("\nMatrix C:\n", C)
    print("Inverse of C:\n", matrix_inverse(C))

    # 7. Транспонування
    print("\nTranspose of A:\n", matrix_transpose(A))

    # 8. Добуток Адамара
    print("\nHadamard product A ⊙ B =\n", hadamard_product(A, B))

    # 9. Норма матриці
    print("\nFrobenius norm of A:", norm(A, 'fro'))
    print("Spectral norm of A:", norm(A, 2))
    print("Max norm of A:", norm(A, np.inf))

    # 10. Базис (лінійно незалежні стовпці)
    D = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [1, 1, 1]], dtype=float)
    print("\nMatrix D:\n", D)
    try:
        print("Basis column indices:", basis(D))
    except Exception as e:
        print("Basis computation error:", e)


if __name__ == "__main__":
    main()
