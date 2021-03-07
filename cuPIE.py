"""cuda-base Poisson Image Editing (cuPIE).
"""

from os import path
import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.linalg
import cupyx.scipy.ndimage
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
from cupyx.scipy.sparse.linalg import lsqr as cupy_lsqr

import scipy.sparse
import scipy.ndimage
from scipy.sparse.linalg import spsolve

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
    
    # data = -cp.ones([5,m*n],cp.float)
    # offsets=cp.array([0,-1,1,-m,m])

    # data[0,:] *= -4    
    # data[2,:] = cp.array(n * ([0] + [-1] * (m-1)))
    # data[1,:] = cp.array(n * ([-1] * (m-1) + [0]))

    # mat_A = cupyx.scipy.sparse.spdiags(data, offsets, m*n, m*n)
    # return mat_A

    data = -cp.ones([5,m*n],cp.float)
    offsets=cp.array([0,-1,1,-m,m])

    data[0,:] *= -4    
    data[2,:] = cp.array(n * ([-1] * (m-1) + [0]))
    data[1,:] = cp.array(n * ([0] + [-1] * (m-1)))

    for dia_idx in range(5):
        data[dia_idx,:] = crop(data[dia_idx,:], offsets[dia_idx], n, m)

    mat_A = cupyx.scipy.sparse.spdiags(data, offsets, m*n, m*n)
    return mat_A

def laplacian_matrix_update_mask(n, m, mask):
    """Generate the Poisson matrix A with a dim of (m*n, m*n)
    Refer to: https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix 

    Args:
        n: This is number of rows
        m: This is number of cols
        mask: cp array with shape = [n,m]

    Returns:
        the Poisson matrix A with a dim of (m*n, m*n) in dia format
    """
    update_mask = cp.array([0]*m + (n-2) * ([0]+ [1]*(m-2)+[0]) + [0]*m).astype(cp.bool)
    update_mask = cp.logical_and(update_mask, (mask.reshape(-1)==0)) #.astype(cp.float)

    data = -cp.ones([5,m*n],cp.float)
    neighbor_update_values = cp.array([1,0,0,0,0])
    offsets=cp.array([0,-1,1,-m,m])

    # Old values
    data[0,:] *= -4    
    data[2,:] = cp.array(n * ([-1] * (m-1) + [0]))
    data[1,:] = cp.array(n * ([0] + [-1] * (m-1)))
    
    # update A
    for dia_idx in range(5):
        data[dia_idx,:] = data[dia_idx,:]*cp.logical_not(update_mask) + update_mask*neighbor_update_values[dia_idx]
        data[dia_idx,:] = crop(data[dia_idx,:], offsets[dia_idx], n, m)

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

def _poisson_edit(source, target, mask, offset):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min
        
    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))
    
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    
    ############################
    # Calculation parts
    ############################
    mat_A = _laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0    
    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        # x = spsolve(mat_A, mat_b)
        # #print(x.shape)
        # x = x.reshape((y_range, x_range))
        # #print(x.shape)

        # x[x > 255] = 255
        # x[x < 0] = 0
        # x = x.astype('uint8')
        # #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # #print(x.shape)

        # target[y_min:y_max, x_min:x_max, channel] = x

    return target

def poisson_edit(source, target, mask, offset):
    """The poisson blending function. 
    Refer to: Perez et. al., "Poisson Image Editing", 2003.

    Args:
        source: a RGB image (cupy ndarray, uint8) with a shape [height, width, 3]
        target: a RGB image (cupy ndarray, uint8) with a shape [height, width, 3]
        mask: a binary mask (cupy ndarray, uint8) with a shape [height, width]
        offset: a tuple of (int,int) describe the geometric transformation of mask

    Returns:
        a RGB image (cupy ndarray) with a shape [height, width, 3]
    """

    y_max, x_max = target.shape[:-1]    # [height, width]
    y_min, x_min = 0, 0
    x_range = x_max - x_min             # target-width
    y_range = y_max - y_min             # target-height

    # Warp affine the source corresponding to the moving mask
    M = cp.eye(3).astype(cp.float32)
    source = cupyx.scipy.ndimage.affine_transform(source,M,[-offset[1],-offset[0],0])[:y_range,:x_range,...]
    
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1

    ############################
    # Calculation parts
    ############################
    # cupyx.scipy.sparse with 'dia' format
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    mat_A = laplacian_matrix_update_mask(y_range, x_range, mask).tocsr()

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        print(type(mat_b), mat_b.shape)

        # x = cupy_lsqr(mat_A, mat_b)
        # x = x.reshape((y_range, x_range))
        
        # x[x > 255] = 255
        # x[x < 0] = 0
        # x = x.astype(cp.uint8)

        # target[y_min:y_max, x_min:x_max, channel] = x

    return target


import time 

if __name__ == '__main__':
    source = cv2.imread("source.png") 
    target = cv2.imread("target.png") 
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE) 
    offset = (-12, 246)

    # CPU-numpy
    start_time = time.time()
    cpu_result = _poisson_edit(target, source, mask, offset)
    # print(cpu_result.getformat())
    elapsed_time = time.time() - start_time
    print(f"CPU time = {elapsed_time}")

    # GPU-cupy
    print("\n---------------\n")
    source = cp.array(source)
    target = cp.array(target)
    mask = cp.array(mask)
    start_time = time.time()
    gpu_result = poisson_edit(target, source, mask, offset)
    # print(gpu_result.getformat())
    elapsed_time = time.time() - start_time
    print(f"CPU time = {elapsed_time}")



    # print(cp.array_equal(gpu_result.todense(), cp.array(cpu_result.todense())))

    # Show results
    # cv2.imshow("cpu_result", cpu_result)
    # cv2.imshow("gpu_result", gpu_result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("Done !")


