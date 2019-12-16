import numpy as np
import matplotlib.pyplot as plt
from os import path
import urllib.request

# a)

def compute_gerschgorin_circles(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Must be square matrix")
    dim = A.shape[0]
    ms = [A[i,i] for i in range(dim)]
    rs = [np.vectorize(abs)(A)[i,:].sum() - abs(A[i,i]) for i in range(dim)]
    return np.array(ms), np.array(rs)

# b)

def gerschgorin_bounds(A):
    ms,rs=compute_gerschgorin_circles(A)
    minimum = min(ms-rs)
    maximum = max(ms+rs)
    return minimum, maximum


# c)

# Falls Matrix nicht im Verzeichnis liegt, von der Website runterladen.
if not path.exists("gerschgorin_50.txt"):
	url = 'https://ins.uni-bonn.de/media/public/courses/WS1920/einfuhrung-in-die-grundlagen-der-numerik/gerschgorin_50.txt'
	urllib.request.urlretrieve(url, 'gerschgorin_50.txt')

A = np.loadtxt('gerschgorin_50.txt', delimiter=',')

if A.shape[0] != A.shape[1]:
    raise ValueError("Must be square matrix")
n = A.shape[0]
colors = plt.cm.rainbow(np.linspace(0, 1, n))

fig, ax = plt.subplots()
for m,r,c in zip(*compute_gerschgorin_circles(A),colors):
    ax.add_patch(plt.Circle((m,0),r,color=c))
ax.autoscale_view()
plt.show()