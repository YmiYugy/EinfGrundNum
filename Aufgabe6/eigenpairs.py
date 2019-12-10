import numpy as np

def power_iteration(A,v_0,max_it):
    # Input:
    #       A       Reelle, diagonalisierbare  n x n Matrix mit 
    #               einem betragsmäßig größten Eigenwert
    #       v_0     Startvektor der Iteration
    #       max_it  Maximale Anzahl an Iterationen
    # Output:
    #       lambda  Approximation an den betragsmäßig größten 
    #               Eigenwert
    #       v       Approximation an den Eigenvektor zum 
    #               betragsmäßig größten Eigenwert
    improvement =   1
    iteration   =   0
    
    v_new       =   v_0
    lambda_new  =   0.
    while improvement > 1e-8 and iteration < max_it:
        iteration   +=  1
        v_old       =   v_new
        lambda_old  =   lambda_new 
        Av_old      =   np.dot(A,v_old)
        v_new       =   1/np.linalg.norm(Av_old) * Av_old
        lambda_new  =   1/np.linalg.norm(v_new) * np.dot(v_new,np.dot(A,v_new))
        improvement =   np.linalg.norm(v_new - v_old)
    
    # Ist das Verfahren konvergiert? 
    if iteration > max_it:
        print("Power-Iteration nach 100 Iterationen nicht konvergiert!")
        return 0., np.zeros(len(A))
    else:
        return lambda_new,v_new

def rank_1_update(A,lam,ev):
    # Input:
    #       A       Reelle n x n Matrix
    #       lam     Eigenwert von A
    #       ev      Eigenvektor von A zum Eigenwert lam
    # Output:
    #       B       Matrix die durch Rang-1-Update von A entsteht
    #               B hat die gleichen Eigenwerte wie A, ausser dass 
    #               der Eigenwert lam auf 0 gesetzt wurde
    B = A - lam * np.kron(ev,ev).reshape(A.shape)
    return B

def compute_first_eigenpairs(A,v_0,m,max_it):
    # Input:        
    #       A       Reelle n x n Matrix mit betragsmäßig strikt geordneten Eigenwerten
    #       v_0     Startvektor der Power-Iterationen
    #       m       Anzahl der Eigenpaare die berechnet werden (maximal len(A) Stück)
    #       max_it  Maximale Iterationszahl pro Power-Iteration
    # Output:
    #       lams    Liste mit den Eigenwerten
    #       evs     Liste mit den Eigenvektoren
    
    B       =   A
    lams    =   []
    evs     =   []
    for i in range(min(m,len(B))):
        # Berechne Eigenpaar zum betragsmäßig größten Eigenwert mit der Power-Iteration
        lam,ev  =   power_iteration(B,v_0,max_it)        
        lams.append(lam)
        evs.append(ev)
        # Führe rang-1-update durch, bei dem der betragsmäßig größte Eigenwert durch 0 ersetzt wird. 
        # Die resultierende Matrix hat ansonsten alle Eigenwerte die B auch hat
        B = rank_1_update(B,lam,ev)

    # Fertig!
    return lams,evs