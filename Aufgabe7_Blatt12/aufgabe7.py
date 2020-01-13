import numpy as np
from scipy.linalg import block_diag
import time
from os import path
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

# e)

# f)
# Falls Matrix nicht im Verzeichnis liegt, von der Website runterladen.
if not path.exists("matrix_qr2.txt"):
	url = 'https://ins.uni-bonn.de/media/public/courses/WS1920/einfuhrung-in-die-grundlagen-der-numerik/matrix_qr2.txt'
	urllib.request.urlretrieve(url, 'matrix_qr2.txt')

A = np.loadtxt('matrix_qr2.txt', delimiter=',')


# Householder-Verfahren
print("Householder-Verfahren. Zur Kontrolle:")

start=time.time()
q_h,r_h = compute_qrdecomp_householder(A)
end = time.time()

print("||Q||₂ =", np.linalg.norm(q_h, ord=2))
print("||Q*Qᵀ - I||F =", np.linalg.norm(q_h @ q_h.T - np.identity(A.shape[0])))
print("||R - np.triu(R)||F =",np.linalg.norm(r_h-np.triu(r_h)))
print("||Q*R - A||F =", np.linalg.norm(q_h@r_h-A))

print("Ausführungsdauer: {:5.3f} Sekunden".format(end-start))
