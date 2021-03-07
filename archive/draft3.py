from os import path
import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.linalg
import cupyx.scipy.ndimage

import scipy.sparse
import scipy.ndimage
from scipy.sparse.linalg import spsolve


def laplacian_matrix(n, m):
    """Generate the Poisson matrix A with a dim of (m*n, m*n)
    Refer to: https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix 

    Args:
        n: This is number of rows
        m: This is number of cols

    Returns:
        the Poisson matrix A with a dim of (m*n, m*n) in dia format
    """
    
    data = -cp.ones([5,m*n],cp.float)
    data[0,:] *= -4    
    data[2,:] = cp.array(n * ([0] + [-1] * (m-1)))
    data[1,:] = cp.array(n * ([-1] * (m-1) + [0]))

    offsets=cp.array([0,-1,1,-m,m])
    mat_A = cupyx.scipy.sparse.spdiags(data, offsets, m*n, m*n)

    return mat_A

def _laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A




nRows = 3
nCols = 5
mask = np.random.randint(2,size=[nRows,nCols])

#CPU test
cpu_a = _laplacian_matrix(nRows,nCols)

for y in range(1, nRows - 1):
    for x in range(1, nCols - 1):
        if mask[y, x] == 0:
            k = x + y * nCols
            cpu_a[k, k] = 1
            cpu_a[k, k + 1] = 0
            cpu_a[k, k - 1] = 0
            cpu_a[k, k + nCols] = 0
            cpu_a[k, k - nCols] = 0

print(cpu_a.getformat())
# print(cpu_a.todense())


#GPU test
print("\n------------\n")
gpu_a = laplacian_matrix(nRows,nCols).tocsc()

for y in range(1, nRows - 1):
    for x in range(1, nCols - 1):
        if mask[y, x] == 0:
            k = x + y * nCols
            gpu_a[k, k] = 1
            gpu_a[k, k + 1] = 0
            gpu_a[k, k - 1] = 0
            gpu_a[k, k + nCols] = 0
            gpu_a[k, k - nCols] = 0

print(gpu_a.getformat())
# print(gpu_a.todense())

print(cp.array_equal(gpu_a.todense(), cp.array(cpu_a.todense())))




