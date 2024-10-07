import numpy as np

FILENAME = 'input.txt'

def read_input(file):
    list_file = list(file)

    A = [[float(elem) for elem in list_file[i].split()] for i in range(1, len(list_file) - 1)]
    b = [float(elem) for elem in list_file[-1].split()]

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

def det_gauss(matrix):
    n = matrix.shape[0]
    Ab = matrix.copy()
    
    swap_count = 0 

    for i in range(n):

        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        
        if Ab[max_row, i] == 0:
            return 0
        
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            swap_count += 1 
        
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] = Ab[j] - factor * Ab[i]
    
    det_value = (-1) ** swap_count * np.prod(np.diag(Ab))
    
    return det_value

def gauss_method_with_pivot_solve(matrix, vector):

    def gauss_method_with_pivot_check(matrix, vector):
        n = matrix.shape[0]

        if matrix.shape[1] != n or vector.shape[0] != n:
            raise ValueError('Matrix must be square and match the dimensions of the vector')
        
    try:
        gauss_method_with_pivot_check(matrix, vector)
    except ValueError as e:
        raise e
    
    if not det(matrix):
        raise ValueError('Matrix must not be singular')

    n = len(vector)
    
    identity_matrix = np.eye(n)
    Ab = np.hstack([matrix, identity_matrix, vector.reshape(-1, 1)])

    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if Ab[max_row, i] == 0:
            raise ValueError("System does not have a unique solution")
        
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        Ab[i] = Ab[i] / Ab[i, i]
        
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
    
    res = np.zeros(n)
    inverse_matrix = np.zeros((n, n))

    for i in range(n - 1, -1, -1):
        res[i] = Ab[i, -1] - np.sum(Ab[i, i + 1 : n] * res[i + 1 : n])
        inverse_matrix[i] = Ab[i, n:n + n] - np.dot(Ab[i, i + 1:n], inverse_matrix[i + 1:n])

    return res, inverse_matrix

try:
    with open(FILENAME, 'r') as file:
        A, b = read_input(file.readlines())
except FileNotFoundError as e:
    raise

try:
    gauss_res = gauss_method_with_pivot_solve(A, b)
    det = det_gauss(A)
    print(*gauss_res)
    print(det)
except ValueError as e:
    raise