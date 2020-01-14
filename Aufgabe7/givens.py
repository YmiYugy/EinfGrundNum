import numpy as np
import time
from math import sqrt


def compute_givens_rotation(A, i, j):
    # Input
    #   A       real matrix, n x n
    #   i       row index
    #   j       column index, i > j
    # Output
    #   G       Matrix of givens rotation for A_ij -> 0
    if A.shape[0] != A.shape[1]:
        raise ValueError("Must be square matrix")
    elif j >= i:
        raise ValueError("i must be greater than j")
    a = float(A[j][j]/sqrt(A[j][j]**2 + A[i][j]**2))
    
    b = float(-A[i][j]/sqrt(A[j][j]**2 + A[i][j]**2))
    Q = np.zeros(A.shape)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            if x == y and x != i and y != j:
                Q[x][y] = 1
            elif (x == y and y == i) or (x == y and y == j):
                Q[x][y] = a
            elif x == i and y == j:
                Q[x][y] = -b
            elif x == j and y == i:
                Q[x][y] = b
            else:
                Q[x][y] = 0
    return Q

def compute_qrdecomp_givens(A):
    # Input
    #   A       real n x n matrix
    # Output
    #   Q       matrix Q from qr decomp
    #   R       matrix R from qr decomp
    gs = []
    r = A
    if A.shape[0] != A.shape[1]:
        raise ValueError("Must be square matrix")
    for j in range(A.shape[0]-1):
        for i in range(j+1, A.shape[0]):
            g = compute_givens_rotation(r, i, j)
            gs.append(g)
            r = g.dot(r)
    Q = np.eye(A.shape[0])
    for g in gs:
        Q = Q.dot(g.T)
    return Q, r



start = time . time ( )
A = np.loadtxt("matrix_qr.txt", delimiter=",")
Q, R = compute_qrdecomp_givens(A)
end = time.time()
print('Execution took {:5.3f} seconds'.format(end-start))
