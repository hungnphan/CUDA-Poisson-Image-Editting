"""cuda-base Poisson Image Editing (cuPIE).
"""

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
        the Poisson matrix A with a dim of (m*n, m*n)
    """
    
    data = -cp.ones([5,m*n],cp.float)
    data[0,:] *= -4    
    data[2,:] = cp.array(n * ([0] + [-1] * (m-1)))
    data[1,:] = cp.array(n * ([-1] * (m-1) + [0]))

    offsets=cp.array([0,-1,1,-m,m])
    mat_A = cupyx.scipy.sparse.spdiags(data, offsets, m*n, m*n)

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
    img = cv2.warpAffine(source,M,(x_range,y_range))
    
    return img

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
    img = cupyx.scipy.ndimage.affine_transform(source, M, [-offset[1] , -offset[0] , 0])[:y_range,:x_range,...]
    
    return img



if __name__ == '__main__':
    source = cv2.imread("source.png") 
    target = cv2.imread("target.png") 
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE) 
    offset = (-12, 246)

    # CPU-numpy
    cpu_result = _poisson_edit(target, source, mask, offset)

    # GPU-cupy
    source = cp.array(source)
    target = cp.array(target)
    mask = cp.array(mask)
    gpu_result = poisson_edit(target, source, mask, offset)
    gpu_result = cp.asnumpy(gpu_result)

    cv2.imshow("cpu_result", cpu_result)
    cv2.imshow("gpu_result", gpu_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Done !")


