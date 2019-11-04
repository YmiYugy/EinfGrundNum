import numpy as np
import matplotlib.pyplot as plt

# Aufgabenteil a)

def evaluate_bspline_recursive(ts,xs,i,p):
    # Input:
    #   ts      Knotenvektor
    #   xs      Punkte an denen N_{i,p} ausgewertet werden soll
    #   i       Nummer des B-Splines, siehe Definition
    #   p       Ordnung des B-Splines, siehe Definition
    # Output:
    #   ys      Funktionswerte fuer x in xs
    if i + p >= len(ts) - 1:
        raise ValueError("B-spline does not exist")
    if p==0:
        return np.array([ts[i] <= x < ts[i+1] for x in xs],dtype=float)
    xs = np.array(xs, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    if ts[i+p] != ts[i]:
        fs = (xs - ts[i])/(ts[i+p] - ts[i])
        ys += fs * evaluate_bspline_recursive(ts,xs,i,p-1) 
    if ts[i+p+1] != ts[i+1]:
        fs = (ts[i+p+1] - xs)/(ts[i+p+1] - ts[i+1])
        ys += fs * evaluate_bspline_recursive(ts,xs,i+1,p-1)
    return ys

# Aufgabenteil b)



# Aufgabenteil c)

# Abgewandelte Funktion aus der Musterlösung 
def trapezregel_mInterval(func,a,b,m):
    trapezregel = lambda f, a, b: (b-a) * (f(a) + f(b)) / 2
    # Splitte das Interval in m Stücke. Die Länge der Teilstücke ist:
    h   =   (b-a)/m 
    # Berechne die Werte der Trapezregel auf den Teilintervallen 
    approx_on_subIntervals  =   [trapezregel(func,a+i*h,a+(i+1)*h) for i in range(m)]
    # Summiere die Approximationen der Teilintegrale 
    approx  =   sum(approx_on_subIntervals)
    return approx

def assemble_grammatrix_bspline_recursive(ts,p):
    # Input:
    #   ts      Knotenvektor
    #   p       Ordnung der B-Splines, siehe Definition
    # Output:
    #   G       Gram Matrix als np.array
    
    m = len(ts) - p - 1

    f = lambda i, x: evaluate_bspline_recursive(ts,[x],i,p)[0]
    inner_product = lambda i, j: trapezregel_mInterval(lambda x: f(i,x)*f(j,x),0,1,50)

    return np.array([[inner_product(i,j) for i in range(m)] for j in range(m)])
