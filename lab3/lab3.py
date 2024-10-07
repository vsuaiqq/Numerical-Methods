import numpy as np

EPS = 1E-9

FILENAME = 'input.txt'

def read_input(file):
    list_file = list(file)

    A = [[float(elem) for elem in list_file[i].split()] for i in range(1, len(list_file) - 1)]
    b = [float(elem) for elem in list_file[-1].split()]

    return np.array(A), np.array(b)

def l2_norm(vector):
    return np.sqrt(np.sum(vector ** 2)) if vector.ndim == 1 else np.sqrt(np.sum(vector ** 2))
    
def matrix_norm(matrix):
    max_sum = 0
    for row in matrix:
        row_sum = sum(abs(elem) for elem in row)
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

def simple_iterations_method_solve(matrix, vector, eps=1e-6, max_iterations=500):

    def simple_iterations_method_check(matrix, vector):
        n = matrix.shape[0]

        if matrix.shape[1] != n or vector.shape[0] != n:
            raise ValueError('Matrix must be square and match the dimensions of the vector')
    
    try:
        simple_iterations_method_check(matrix, vector)
    except ValueError as e:
        raise e
    
    n = len(vector)

    res = np.zeros_like(vector)
    res_prev = np.zeros_like(vector)

    B = np.zeros_like(matrix)
    c = np.zeros_like(vector)

    for i in range(n):
        for j in range(n):
            if i != j:
                B[i, j] = -matrix[i, j] / matrix[i, i]
        c[i] = vector[i] / matrix[i, i]

    for iter in range(max_iterations):
        res = B.dot(res) + c

        if l2_norm(res - res_prev) < eps:
            return res, iter, matrix_norm(B)
        
        res_prev = res

    raise ValueError('The method of simple iterations did not converge')

def seidel_method_solve(matrix, vector, eps=1e-6, max_iterations=500):

    def seidel_method_check(matrix, vector):
        n = matrix.shape[0]

        if matrix.shape[1] != n or vector.shape[0] != n:
            raise ValueError('Matrix must be square and match the dimensions of the vector')
    
    try:
        seidel_method_check(matrix, vector)
    except ValueError as e:
        raise e
    
    n = len(vector)

    res_prev = np.zeros_like(vector)

    for iter in range(max_iterations):
        res = np.copy(res_prev)

        for i in range(n):
            sum1 = sum(matrix[i][j] * res[j] for j in range(i))
            sum2 = sum(matrix[i][j] * res[j] for j in range(i + 1, n))
            res[i] = (vector[i] - sum1 - sum2) / matrix[i, i]

        if l2_norm(res - res_prev) < eps:
            return res, iter

        res_prev = res

    raise ValueError('The method of seidel did not converge')

try:
    with open(FILENAME, 'r') as file:
        A, b = read_input(file.readlines())
except FileNotFoundError as e:
    raise

try:
    simple_iterations_res = simple_iterations_method_solve(A, b, EPS)
    print(*simple_iterations_res)
except ValueError as e:
    raise

try:
    seidel_res = seidel_method_solve(A, b, EPS)
    print(*seidel_res)
except ValueError as e:
    raise