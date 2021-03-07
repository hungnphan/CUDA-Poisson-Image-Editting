import numpy as np
import cupy as cp
import cv2
import cupyx.scipy.sparse
import cupyx.scipy.linalg
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage

offset_x_cv = -12 #
offset_y_cv = 246 #
offset_z_cv = 0 #

img = cv2.imread("target.png") 
offset = (offset_x_cv, offset_y_cv)
M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
warp_cv = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
print(type(warp_cv), warp_cv.dtype)
print(warp_cv.shape)

img = cv2.imread("target.png") 
M = np.eye(3).astype(np.float32)
offset = [-offset_y_cv, -offset_x_cv, offset_z_cv]
warp_sci = scipy.ndimage.affine_transform(img,M,offset)
print(type(warp_sci), warp_sci.dtype)
print(warp_sci.shape)

print()
# print(np.sum(np.abs(warp_cv.astype(int)-warp_sci.astype(int))))
print(np.array_equal(warp_cv, warp_sci))
print(np.sum(warp_cv[0].astype(int)-warp_sci[0].astype(int)),\
    np.sum(warp_cv[1].astype(int)-warp_sci[1].astype(int)),\
    np.sum(warp_cv[2].astype(int)-warp_sci[2].astype(int)))
print(np.array_equal(warp_cv[0], warp_sci[0]),\
    np.array_equal(warp_cv[1], warp_sci[1]),\
    np.array_equal(warp_cv[2], warp_sci[2]))


cv2.imshow("warp_cv", warp_cv)
cv2.imshow("warp_sci", warp_sci)
cv2.waitKey(0)






# print((warp_cv==warp_sci).all())
# print((warp_cv[0]==warp_sci[0]).all(),\
#     (warp_cv[1]==warp_sci[1]).all(),\
#     (warp_cv[2]==warp_sci[2]).all())
