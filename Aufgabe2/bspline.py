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
