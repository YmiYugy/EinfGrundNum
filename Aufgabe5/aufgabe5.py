import numpy as np
from os import path
import urllib.request

# Falls Matrix nicht im Verzeichnis liegt, von der Website runterladen.
if not path.exists("matrix_power.txt"):
	url = 'https://ins.uni-bonn.de/media/public/courses/WS1920/einfuhrung-in-die-grundlagen-der-numerik/matrix_power.txt'
	urllib.request.urlretrieve(url,'matrix_power.txt')

A = np.loadtxt('matrix_power.txt',delimiter=',')

# Augabenteil a)
def power_iteration(A,v0, max_it):
	v = np.array(v0)
	A = np.array(A)
	for i in range(max_it):
		tmp = v
		v = A @ v 
		v /= np.linalg.norm(v)
		if np.linalg.norm(v-tmp)<1e-8:
			return v
			
# Aufgabenteil b)
print("b)\nv =")
v = power_iteration(A,np.ones(A.shape[0]),  10000)
print(v)

# Aufgabenteil c)
def rank_1_update(A,lam,ev):
	return A-lam*np.outer(ev,ev)
	
# Aufgabenteil d)

def compute_first_eigenpairs(A,v_0,m,max_it):
	lams=[]
	evs=[]
	for i in range(m):
		ev = power_iteration(A,v_0,max_it)
		if ev is None:
			break
		lam = np.dot(A@ev,ev)/np.dot(ev,ev)
		evs.append(ev)
		lams.append(lam)
		A = rank_1_update(A, lam, ev)
	return lams,evs
	
# Aufgabenteil e)
print("\ne)")
print(compute_first_eigenpairs(A,np.ones(A.shape[0]),A.shape[0],10000)[0])
		