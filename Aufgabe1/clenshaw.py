import numpy as np
import matplotlib.pyplot as plt

# Aufgabenteil a)
def clenshaw(alphas, xs):
    # Input:
    #   alphas  Liste / np.array mit den Koeffizienten der
    #           Entwicklung in der Tschebyscheff Basis
    #   xs      Liste / np.array mit den x-Werten an denen
    #           ausgewertet werden soll
    #
    # Output:
    #   ys      Liste mit den y-Werten

    alphas = list(alphas)
    xs = np.array(xs)

    if len(alphas) < 3:
        alphas += (3 - len(alphas))*[0]

    def betas(x):
        beta2 = alphas[-1]
        beta1 = alphas[-2] + 2*x*beta2
        _alphas = alphas[1:-2]
        while _alphas:
            tmp = _alphas.pop() + 2*x*beta1 - beta2
            beta2 = beta1
            beta1 = tmp
        return alphas[0] + x*beta1 - beta2
    return list(map(betas, xs))


# Aufgabenteil b)
def plotChebyshev(n):
    xs = np.linspace(-1, 1, num=200)
    yss = [clenshaw([1 if i == j else 0 for i in range(j+1)], xs) for j in range(n)]
    plt.title(f"Die ersten {n} Chebyshev-Polynome")
    for i, ys in enumerate(yss):
        plt.plot(xs, ys, label=f"n={i}")
    plt.legend(loc="best")
    plt.savefig("ChebyshevSample6.pdf", format="pdf")
    plt.clf()

plotChebyshev(6)


# Aufgabenteil c)
def recursion_3term_iterative(F0, F1, beta, gamma, n, xs):
    # Input:
    #   F0       lambda function, Startwert 0
    #   F1       lambda function, Startwert 1
    #   beta     Parameter aus Rekursion
    #   gamma    Zweiter Parameter aus Rekursion
    #   n        Ordnung der Rekursion
    #   xs       Punkte an denen ausgewertet werden soll
    # Output:
    #   ys       Liste der Werte Fn(x) fuer x in xs
    xs0 = [F0(x) for x in xs]
    if n == 0:
        return xs0
    xs1 = [F1(x) for x in xs]
    if n == 1:
        return xs1
    for i in range(n-1):
        xs2 = [beta * x1 + gamma * x0 for x0, x1 in zip(xs0, xs1)]
        xs0 = xs1
        xs1 = xs2
    return xs2


def pn_tschebyscheff_iterative(alphas, xs):
    # Input:
    #   alphas      Koeffizienten aus der Entwicklung
    #   xs          Punkten an denen ausgewetet werden soll
    #
    # Output:
    #   ys          Funktionswerte an den Punkten x aus xs

    def tschebyscheff():
        # Generiert die Tschebyscheff-Polynome ausgewertet an den Stellen xs
        xs0 = [1 for x in xs]
        yield xs0
        xs1 = [x for x in xs]
        yield xs1
        while True:
            xs2 = [2*x*x1 - x0 for x, x0, x1 in zip(xs, xs0, xs1)]
            yield xs2
            xs0 = xs1
            xs1 = xs2

    results = [0 for x in xs]
    for ts, alpha in zip(tschebyscheff(), alphas):
        results = [r+alpha*t for r, t in zip(results, ts)]
    return results

def chebyshevError(alphas, xs):
    clenshaw_np = np.array(clenshaw(alphas,xs))
    tschebyscheff_np = np.array(pn_tschebyscheff_iterative(alphas,xs))
    ys = abs(clenshaw_np-tschebyscheff_np)
    plt.title("Absoluter Fehler bei der Auswertung des Polynoms")
    plt.plot(xs, ys)
    plt.savefig("Absoluter Fehler.pdf", format="pdf")
    plt.clf()

chebyshevError([0, 0, 2/3, 0, 4/14, 0, 23/96], np.linspace(0, 1, num=100))
