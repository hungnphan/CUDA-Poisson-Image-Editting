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



# Predefined params
nRows = 23
nCols = 25
mask = np.random.randint(2,size=[nRows,nCols])
# mask = cv2.imread("target_mask.png", cv2.IMREAD_GRAYSCALE)
# nRows, nCols = mask.shape

##############################################
# CPU work
##############################################
cpu_a = _laplacian_matrix(nRows,nCols)
print(cpu_a.getformat()) # lil
# cpu_laplacian = cpu_a.tocsc()

# set the region outside the mask to identity   
y_range = nRows
x_range = nCols
for y in range(1, y_range - 1):
    for x in range(1, x_range - 1):
        if mask[y, x] == 0:
            k = x + y * x_range
            cpu_a[k, k] = 1
            cpu_a[k, k + 1] = 0
            cpu_a[k, k - 1] = 0
            cpu_a[k, k + x_range] = 0
            cpu_a[k, k - x_range] = 0



##############################################
# GPU work
##############################################
mask = cp.array(mask)
print("\n------------\n")
gpu_a = laplacian_matrix(nRows,nCols)
print(gpu_a.getformat()) # diag
# gpu_laplacian = gpu_a.tocsc()

# set the region outside the mask to identity 
# create update mask for indices needed to change
update_mask = cp.array([0]*nCols + (nRows-2) * ([0]+ [1]*(nCols-2)+[0]) + [0]*nCols).astype(cp.bool)
update_mask = cp.logical_and(update_mask, (mask.reshape(-1)==0)) #.astype(cp.float)

def crop(row, k, numRows, numCols):
    # row is cp array
    dim = numRows*numCols
    # assert(row.shape==[dim])
    if k == 0:
        return row
    elif k > 0:
        row[k:] = row[:dim-k]
        row[:k] = 0
        return row
    elif k < 0:
        k = -k
        row[:-k] = row[k:]
        row[-k:] = 0
        return row

n, m = nRows, nCols
data = -cp.ones([5,m*n],cp.float)
neighbor_update_values = cp.array([1,0,0,0,0])
offsets=cp.array([0,-1,1,-m,m])

# old A
data[0,:] *= -4    
data[2,:] = cp.array(n * ([-1] * (m-1) + [0]))
data[1,:] = cp.array(n * ([0] + [-1] * (m-1)))

# update A
for dia_idx in range(5):
    data[dia_idx,:] = data[dia_idx,:]*cp.logical_not(update_mask) + update_mask*neighbor_update_values[dia_idx]
    data[dia_idx,:] = crop(data[dia_idx,:], offsets[dia_idx], n, m)

mat_A_2 = cupyx.scipy.sparse.spdiags(data, offsets, m*n, m*n)
 
#############





##############################################
# Result check
##############################################
print("\n------------\n")
# print(mask.reshape([nRows,nCols]).astype(cp.bool))
print("check matrix a:\t", 
    cp.array_equal(gpu_a.todense(), cp.array(cpu_a.todense())))
print("check matrix a on GPU ONLY:\t", 
    cp.array_equal(cp.array(cpu_a.todense()), mat_A_2.todense()))
# print("check matrix laplacian:\t", 
#     cp.array_equal(gpu_laplacian.todense(), cp.array(cpu_laplacian.todense())))

# print(cpu_a.todense())
# print()
# print(mat_A_2.todense())


