import numpy as np
from scipy.linalg import get_lapack_funcs
from scipy import sparse

def diffMatrix(arr, d):
    diff = arr[1:] - arr[:-1]
    if d > 1:
        diff = diffMatrix(diff, d-1)
    return diff

def ddmat_sp(x, d):
    x = np.array(x)
    m = len(x)
    D = sparse.eye(m).tocsr()
    if d>0:
        dx = x[d:] - x[:m-d]
        V = sparse.diags(1 / dx, 0).tocsr()
        
        D = V.dot(diffMatrix(ddmat_sp(x, d - 1), 1))
    return D

def ddmat(x, d):
    x = np.array(x)
    m = len(x)
    D = np.eye(m)
    if d>0:
        dx = x[d:] - x[:m-d]
        V = np.diag(1 / dx, 0)
        D = V.dot(diffMatrix(ddmat(x, d - 1), 1))
    return D

def robust_smooth_sp(array, Warray, x, s, d, iterations=1, axis=0):
    shape = list(array.shape)
    t = list(range(array.ndim))
    t.remove(axis)
    t = [axis, ] + t
    array  =  array.transpose(t).reshape(shape[axis], -1).T
    Warray = Warray.transpose(t).reshape(shape[axis], -1).T
    
    mask = np.sum(Warray, axis = 1) > 0.000001 #, axis=1) #np.all(array > 0.00001, axis=1) | 

#     w = Warray[mask]
    good_array = array[mask]
    
    
    D_sp = ddmat_sp(x, d)
    DTD_sp = D_sp.T.dot(D_sp) * s

    tmp = np.sqrt(1+16*s)
    h   = np.sqrt(1+tmp)/np.sqrt(2)/tmp
    bottom = np.sqrt(1-h)

    wnew = Warray[mask]
    good_values = wnew > 0
    
    dtd_center = DTD_sp.diagonal(0)[None] 
    if d == 1:
        left_upper = DTD_sp.diagonal(-1)
    else:
        left_upper = np.zeros((d, dtd_center[0].shape[0]))
        for i in range(1, d+1):
            left_upper[i-1, d+1 -i:] = DTD_sp.diagonal(d + 1 - i)
            
    for i in range(iterations):
        right = (good_array * wnew)
        left_center = (dtd_center + wnew)
        if d == 1:
            ptsv, = get_lapack_funcs(('ptsv',), (left_center[0], right[0]))
            xs = [ptsv(left_center[i], left_upper, right[i], False, False,False)[2] for i in range(len(left_center))]
            xs = np.vstack(xs)
        else:
            pbsv, = get_lapack_funcs(('pbsv',), (left_center[0], right[0]))
            xs = [pbsv(np.vstack([left_upper, left_center[[i]]]), right[i], lower=False, overwrite_ab=False, overwrite_b=False)[1] for i in range(len(left_center))]
            xs = np.vstack(xs)

        diff = (xs - good_array)
        residuals = diff[good_values]
        
        MAD = np.median(abs(residuals - np.median(residuals)))
        MAD = np.maximum(MAD, 1e-6)[...,None]
        u = abs(diff / (1.4826*MAD) / bottom)
        wnew = (1 - (u / 4.685) ** 2)**2 * ((u/4.685)<1)
        wnew[~good_values] = 0
 
    new_shape = shape.copy()
    del shape[axis]
    new_shape = tuple([new_shape[axis], ] + shape)

    t = list(range(1, len(new_shape)))
    t.insert(axis, 0)

    array[mask] = xs
    Warray[mask] = wnew

    array = array.T.reshape(new_shape).transpose(t)
    Warray = Warray.T.reshape(new_shape).transpose(t)

    return array, Warray

def diffMatrix(arr, d):
    arr = np.array(arr)
    diff = arr[1:] - arr[:-1]
    if d > 1:
        diff = diffMatrix(diff, d-1)
    return diff

def ddmat(x, d):
    x = np.array(x)
    m = len(x)
    D = np.eye(m)
    if d>0:
        dx = x[d:] - x[:m-d]
        V = np.diag(1 / dx)
        D = V.dot(diffMatrix(ddmat(x, d - 1), 1))
    return D

def diffMatrix(arr, d):
    # arr = np.array(arr)
    diff = arr[1:] - arr[:-1]
    if d > 1:
        diff = diffMatrix(diff, d-1)
    return diff

def tridiagonal_solve(left, right):
    upper  = np.diagonal(left, offset=-1, axis1=1, axis2=2)
    center = np.diagonal(left, offset= 0, axis1=1, axis2=2)

    ptsv, = get_lapack_funcs(('ptsv',), (center[0], right[0]))
    xs = [ptsv(center[i], upper[i], right[i], False, False,False)[2] for i in range(len(left))]
    xs = np.vstack(xs)
    
    return xs

def cholesky_banded_solve(left, right, bandwidth):
    
    a1 = np.zeros((len(left), bandwidth+1, left[0].shape[0]))
    for i in range(bandwidth+1):
        a1[:, bandwidth - i, i:] = np.diagonal(left, offset=-i, axis1=1, axis2=2)
    pbsv, = get_lapack_funcs(('pbsv',), (a1[0], right[0]))
    xs = [pbsv(a1[i], right[i], lower=False, overwrite_ab=False, overwrite_b=False)[1] for i in range(len(left))]
    xs = np.vstack(xs)
    return xs
    
        
def robust_smooth(array, Warray, x, s, d, iterations=1, axis=0):
    
    shape = list(array.shape)
    t = list(range(array.ndim))
    t.remove(axis)
    t = [axis, ] + t
    array  =  array.transpose(t).reshape(shape[axis], -1).T
    Warray = Warray.transpose(t).reshape(shape[axis], -1).T
    
    mask = np.sum(Warray, axis = 1) > 0.000001 #, axis=1) #np.all(array > 0.00001, axis=1) | 
    left = np.zeros((mask.sum(), shape[axis], shape[axis]))
    diag_x_ind, diag_y_ind = np.arange(shape[axis]), np.arange(shape[axis])
    left[:, diag_x_ind, diag_y_ind] = Warray[mask]
    D = ddmat(x, d)
    
#     D = diffMatrix(np.eye(shape[axis]), d)
    DTD = D.T.dot(D) * s
    DTD = DTD[None, ...]

    tmp = np.sqrt(1+16*s)
    h   = np.sqrt(1+tmp)/np.sqrt(2)/tmp
    bottom = np.sqrt(1-h)
    
    wnew = Warray[mask]
    # iii = 86
    good_values = wnew>0
    
    good_array = array[mask]
    
    
    for i in range(iterations):
        right = good_array *  wnew 
#         smoothed = np.linalg.solve(left + DTD, right)        
        if d == 1:
            smoothed = tridiagonal_solve(left + DTD, right.copy())
        else:
            smoothed = cholesky_banded_solve(left + DTD, right.copy(), d)
        
        diff = (smoothed - good_array)
#         if i < (iterations-2):
#             lower_value_mask = diff < 0
#             good_array[lower_value_mask] = smoothed[lower_value_mask]
#             diff = (smoothed - good_array)
        
        residuals = diff[good_values]
        
        MAD = np.median(abs(residuals - np.median(residuals)))

    
        MAD = np.maximum(MAD, 0.00001)[...,None]
        u = abs(diff / (1.4826*MAD) / bottom)
        wnew = (1 - (u / 4.685) ** 2)**2 * ((u/4.685)<1)
        left[:, diag_x_ind, diag_y_ind] = wnew

    new_shape = shape.copy()
    del shape[axis]
    new_shape = tuple([new_shape[axis], ] + shape)

    t = list(range(1, len(new_shape)))
    t.insert(axis, 0)

    array[mask] = smoothed
    Warray[mask] = wnew

    array = array.T.reshape(new_shape).transpose(t)
    Warray = Warray.T.reshape(new_shape).transpose(t)

    return array, Warray