import scipy as sp
import numpy as np


def symmetric_matrix(A):
    return (A + A.T) / 2

# generate 5 different size x 2types {sparse, symmetric} square matrices
densities=np.linspace(0.1,1.0,10)
sides=[6, 8, 10, 12, 14]

for si in sides:
    for di in densities:
        S = sp.sparse.random(2**si, 2**si, density=di,dtype=np.float32).toarray()
        P = symmetric_matrix(S)
        np.savetxt(f"S_{si}_{di:.1f}.txt", S, fmt="%.6f", delimiter=" ")
        np.savetxt(f"ST_{si}_{di:.1f}.txt", S.T, fmt="%.6f", delimiter=" ")
        np.savetxt(f"P_{si}_{di:.1f}.txt", P, fmt="%.6f", delimiter=" ")







