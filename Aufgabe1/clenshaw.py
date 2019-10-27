# Aufgabenteil c)

def recursion_3term_iterative(F0,F1,beta,gamma,n,xs):
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
    if n==0:
        return xs0
    xs1 = [F1(x) for x in xs]
    if n==1:
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
