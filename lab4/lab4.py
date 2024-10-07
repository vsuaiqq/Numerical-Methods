import numpy as np

EPS = 1E-9

FILENAME = 'input.txt'

def read_input(file):
    list_file = list(file)

    A = [[float(elem) for elem in list_file[i].split()] for i in range(1, len(list_file))]

    return np.array(A)

def max_offdiag(matrix):
    n = matrix.shape[0]

    max_offdiag_val = 0
    max_offdiag_str, max_offdiag_col = 0, 1
    
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > abs(max_offdiag_val):
                max_offdiag_val = matrix[i, j]
                max_offdiag_str, max_offdiag_col = i, j

    return max_offdiag_val, max_offdiag_str, max_offdiag_col


def jacobi_method_solve(matrix, eps=1e-6, max_iterations=500):

    def jacobi_method_check(matrix):

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Matrix must be square and match the dimensions of the vector')

        if not np.array_equal(matrix, matrix.T):
            raise ValueError('Matrix should be symmetric')

    try:
        jacobi_method_check(matrix)
    except ValueError as e:
        raise e
    
    n = matrix.shape[0]

    V = np.eye(n)
    
    iterations = 0

    for _ in range(max_iterations):
        max_offdiag_val, max_offdiag_str, max_offdiag_col = max_offdiag(matrix)

        if abs(max_offdiag_val) < eps:
            break

        if matrix[max_offdiag_str, max_offdiag_str] == matrix[max_offdiag_col, max_offdiag_col]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * matrix[max_offdiag_str, max_offdiag_col] / (matrix[max_offdiag_str, max_offdiag_str] - matrix[max_offdiag_col, max_offdiag_col]))

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        J = np.eye(n)
        J[max_offdiag_str, max_offdiag_str] = cos_phi
        J[max_offdiag_col, max_offdiag_col] = cos_phi
        J[max_offdiag_str, max_offdiag_col] = -sin_phi
        J[max_offdiag_col, max_offdiag_str] = sin_phi

        matrix = J.T @ matrix @ J
        V = V @ J

        iterations += 1

    eigenvalues = np.diag(matrix)
    eigenvectors = V

    return eigenvalues, eigenvectors, iterations

try:
    with open(FILENAME, 'r') as file:
        A = read_input(file.readlines())
except FileNotFoundError as e:
    raise

try:
    jacobi_res = jacobi_method_solve(A, EPS)
    print(*jacobi_res)
except ValueError as e:
    raise