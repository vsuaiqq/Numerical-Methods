import numpy as np

FILENAME = 'input.txt'

def read_input(file):
    list_file = list(file)

    A = [[int(elem) for elem in list_file[i].split()] for i in range(1, len(list_file) - 1)]
    b = [int(elem) for elem in list_file[-1].split()]

    return np.array(A), np.array(b)

def det(matrix):
    if matrix.shape[0] == 1:
        return matrix[0]
    elif matrix.shape[0] == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    res = 0
    for i in range(matrix.shape[0]):
        res += matrix[0][i] * ((-1) ** (0 + i)) * det(np.delete(np.delete(matrix, 0, axis=0), i, axis=1))
    return res
    
def tridiagonal_matrix_solve_check(matrix, vector):
    n = matrix.shape[0]

    if matrix.shape[0] != n or matrix.shape[1] != n or vector.shape[0] != n:
        raise ValueError('Matrix must be square and match the dimensions of the vector')

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i][j] != 0:
                raise ValueError('Matrix must be tridiagonale')
    
    if any(matrix[i][i] == 0 for i in range(n)):
        raise ValueError('Main diagonal must be non zero')
        
    if not det(matrix):
        raise ValueError('Matrix must not be singular')
    
    under_diag = [0] + [matrix[i][i - 1] for i in range(1, n)]
    main_diag = [matrix[i][i] for i in range(n)]
    upper_diag = [matrix[i][i + 1] for i in range(n - 1)] + [0]
    n = len(main_diag)
    for i in range(n):
        sum_diag = 0
        if i > 0:
            sum_diag = abs(under_diag[i])
        if i < n - 1:
            sum_diag = abs(upper_diag[i])
        if abs(main_diag[i]) < sum_diag:
            raise ValueError('Matrix must be diagonally dominant')

def tridiagonal_matrix_solve(matrix, vector):
    try:
        tridiagonal_matrix_solve_check(matrix, vector)
    except ValueError as e:
        raise e

    n = matrix.shape[0]

    under_diag = [0] + [matrix[i][i - 1] for i in range(1, n)]
    main_diag = [matrix[i][i] for i in range(n)]
    upper_diag = [matrix[i][i + 1] for i in range(n - 1)] + [0]

    alpha = [0] * n
    beta = [0] * n

    alpha[1] = -upper_diag[0] / main_diag[0]
    beta[1] = vector[0] / main_diag[0]

    for i in range(1, n - 1):
        denominator = main_diag[i] + under_diag[i] * alpha[i]
        alpha[i + 1] = -upper_diag[i] / denominator
        beta[i + 1] = (vector[i] - under_diag[i] * beta[i]) / denominator

    res = [0] * n
    res[-1] = (vector[-1] - under_diag[-1] * beta[-1]) / (main_diag[-1] + under_diag[-1] * alpha[-1])

    for i in range(n - 2, -1, -1):
        res[i] = alpha[i + 1] * res[i + 1] + beta[i + 1]

    return res

try:
    with open(FILENAME, 'r') as file:
        A, b = read_input(file.readlines())
except FileNotFoundError as e:
    raise

try:
    x = tridiagonal_matrix_solve(A, b)
    print(*x)
except ValueError as e:
    raise