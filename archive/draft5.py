import time
from os import path
import cupy as cp
import cupyx.scipy.sparse
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve

import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as numpy_spsolve


N = 2000
k = 1

A_ = np.random.randint(1,1000,[N,N]).astype(np.float)
A = csc_matrix(A_, dtype=float)
B_ = np.random.randint(1,1000,[N,k]).astype(np.float)
# B = csc_matrix(B, dtype=float)

start_time = time.time()
x = numpy_spsolve(A, B_)
elapsed_time = time.time() - start_time
print(f"CPU time = {elapsed_time}")



A = cupyx.scipy.sparse.csc_matrix(cp.array(A_), dtype=float)
B = cp.array(B_)
# B = cupyx.scipy.sparse.csc_matrix(B, dtype=float)

start_time = time.time()
x = cupy_spsolve(A, B)
elapsed_time = time.time() - start_time
print(f"GPU time = {elapsed_time}")

