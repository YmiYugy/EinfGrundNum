import numpy as np
from scipy.linalg import block_diag
import time
from os import path
from math import sqrt
import urllib.request

# a)
def compute_householder_matrix_reduced(v):
    # Input:
    #   v       vector of length k
    # Output:
    #   Reduced Householder matrix

    # Anmerkung: In der Aufgabe steht einerseits, dass H=I-2vv^T sein soll, andererseits, dass 
    # es sich hierbei um die Spiegelung von v auf einen Einheitsvektor handeln soll. Ich habe
    # mich für letzteres entschieden.
    w = np.array(v)

    if w.shape!=(w.size,):
        raise ValueError()

    w[0] += np.sign(w[0])*np.linalg.norm(w)
    w = w / np.linalg.norm(w)

    return np.identity(w.size) - np.outer(2*w, w)

# b)
def compute_householder_matrix_full(v,m):
    # Input:
    #   v       Vector of size n
    #   m       slicing index
    # Output:
    #   Full Householder matrix

    if not (0<=m<v.size):
        raise ValueError()

    return block_diag(np.identity(m),compute_householder_matrix_reduced(v[m:]))

# c)
def compute_qrdecomp_householder(A):
    n = A.shape[0]
    if A.shape != (n,n):
        raise ValueError()

    Q = np.identity(n)
    R = np.array(A, dtype=float)
    for i in range(n):
        v = R[:,i]
        H = compute_householder_matrix_full(v,i)
        R = H @ R
        Q = Q @ H
    return (Q,R)

# d)
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


# e)
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

# f)
# Falls Matrix nicht im Verzeichnis liegt, von der Website runterladen.
matrix_file = "matrix_qr.txt" # Alternativ: matrix_qr2.txt

if not path.exists(matrix_file):
	url = 'https://ins.uni-bonn.de/media/public/courses/WS1920/einfuhrung-in-die-grundlagen-der-numerik/'+matrix_file
	urllib.request.urlretrieve(url, matrix_file)

A = np.loadtxt(matrix_file, delimiter=',')


# Householder-Verfahren
print("Householder-Verfahren")

start=time.time()
q_h,r_h = compute_qrdecomp_householder(A)
end = time.time()

print("Ausführungsdauer: {:5.3f} Sekunden".format(end-start))
print("Zur Kontrolle:")
print("||Q||₂ =", np.linalg.norm(q_h, ord=2))
print("||Q*Qᵀ - I||F =", np.linalg.norm(q_h @ q_h.T - np.identity(A.shape[0])))
print("||R - np.triu(R)||F =",np.linalg.norm(r_h-np.triu(r_h)))
print("||Q*R - A||F =", np.linalg.norm(q_h@r_h-A))

print("Givens-Rotation")

start=time.time()
q_g,r_g = compute_qrdecomp_givens(A)
end = time.time()

print("Ausführungsdauer: {:5.3f} Sekunden".format(end-start))
print("Zur Kontrolle:")
print("||Q||₂ =", np.linalg.norm(q_g, ord=2))
print("||Q*Qᵀ - I||F =", np.linalg.norm(q_g @ q_g.T - np.identity(A.shape[0])))
print("||R - np.triu(R)||F =",np.linalg.norm(r_g-np.triu(r_g)))
print("||Q*R - A||F =", np.linalg.norm(q_g@r_g-A))