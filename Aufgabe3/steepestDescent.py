import numpy as np
def steepestDescent(A,b,eps,imax):
    x = np.zeros(A.shape[0],dtype=float)

    for i in range(imax):
        r = A @ x - b
        if np.linalg.norm(r) < eps:
            print("Nach {} Schritten terminiert".format(i))
            return x
        q = r / np.linalg.norm(r)
        l = np.dot(-r, q) / np.dot(A @ q, q)
        x += l*q
    print("Abgebrochen nach {} Schritten".format(imax))

A = np.loadtxt("matrix_500.txt",delimiter=',')
b = np.loadtxt("rhs_500.txt",delimiter=',')
print(steepestDescent(A, b, 1e-7, 10000))
