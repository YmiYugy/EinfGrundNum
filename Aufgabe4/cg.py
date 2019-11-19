import numpy as np
import matplotlib.pyplot as plt

def solve_system_cg(A, b, x_0):
    # Input:
    #   A           System-Matrix
    #   b           rechte Seite
    # x_0           Startvektor
    # Output:
    #   x           Approximation an die Loesung von Ax_*=b
    #   r_k_norms   Liste der Residuen aller Iterationen
    r_k_norms = []
    n = len(b)
    x = x_0
    r= np.dot(A, x) - b
    d = - r
    r_k_n_s = np.dot(r,r)
    for i in range (n):
        Ad = np.dot(A, d)
        alpha = r_k_n_s / np.dot(d, Ad)
        x += alpha * d
        r += alpha * Ad
        r_k_n_s_plus1 = np.dot(r,r)
        beta = r_k_n_s_plus1 / r_k_n_s
        r_k_n_s = r_k_n_s_plus1
        r_k_norms.append(np.sqrt(r_k_n_s))
        if np.sqrt(r_k_n_s) < 1e-8:
            break
        d = beta * d - r
    return (x, r_k_norms)

if __name__ == "__main__":
        A = np.loadtxt("laplacian_200.txt", delimiter=',')
        (x, res) = solve_system_cg(A, np.array([x for x in range(200)]), np.zeros(200))
        #(x, res) = solve_system_cg(np.eye(10), np.zeros(10), np.random.random(10))
        plt.plot(res)
        plt.show()