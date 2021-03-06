"""
    cuda-base Poisson Image Editing (cuPIE).
"""

from os import path
import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.linalg

import scipy.sparse
from scipy.sparse.linalg import spsolve


def _laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    n = nRows
    m = nCols
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A

def laplacian_matrix(n, m):
    """Generate the Poisson matrix A with a dim of (m*n, m*n)

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    n = nRows
    m = nCols
    """
    
    data = -cp.ones([5,m*n],cp.float)
    data[0,:] *= -4    
    data[2,:] = cp.array(n * ([0] + [-1] * (m-1)))
    data[1,:] = cp.array(n * ([-1] * (m-1) + [0]))

    offsets=cp.array([0,-1,1,-m,m])
    mat_A = cupyx.scipy.sparse.spdiags(data, offsets, m*n, m*n)

    return mat_A




# # Debug function whether correct or not
is_passed = True
row_bug, col_bug = -1, -1
for rows in range (3,30):
    for cols in range (3,30):
        arr_cpu = cp.array(_laplacian_matrix(rows,cols).todense())
        arr_gpu = laplacian_matrix(rows,cols).todense()

        if not cp.array_equal(arr_cpu, arr_gpu):
            is_passed = False
            row_bug = rows
            col_bug = cols
            break
    if not is_passed:
        break
print(is_passed)
print(row_bug, col_bug)




# rows = 3
# cols = 3

# arr_cpu = _laplacian_matrix(rows,cols)
# print(arr_cpu.todense().shape)
# print(arr_cpu.todense())
# print()

# arr_gpu = laplacian_matrix(rows,cols)
# # print(arr_gpu.getformat())
# print(arr_gpu.todense().shape)
# print(arr_gpu.todense())



