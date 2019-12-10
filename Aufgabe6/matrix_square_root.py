from eigenpairs import compute_first_eigenpairs
import numpy as np
import itertools

def orthonormalize_gram_schmidt(vs):
    # INPUT:
    #       vs  Liste von Vektoren
    # OUTPUT:
    #       ws  Liste der orthonormalisierten Vektoren
    output = np.array(vs)
    for counter, value in enumerate(output):
        for i in range(counter):
            value -= np.inner(output[i], vs[counter]) * output[i]
        value *= 1/np.linalg.norm(value)
    return output

def compute_root_of_matrix(A):
    # INPUT:
    #       A   reelle spd Matrix
    # OUTPUT:
    #       B   Wurzel von A
    lams, evs = compute_first_eigenpairs(A, [1 for _ in range(len(A))], len(A), 1000)
    vs = orthonormalize_gram_schmidt(evs)
    u = np.matrix(vs).T
    _d = np.sqrt(np.diag(np.diag(u.H * A * u)))
    b = u * _d * u.H
    return b


if __name__ == "__main__":
    A = np.loadtxt("matrix_root.txt", delimiter=',')
    B = compute_root_of_matrix(A)
    print(np.linalg.norm(A-np.dot(B,B), ord='fro'))