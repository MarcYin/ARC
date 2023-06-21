import numpy as np
# from numba import jit
from jax import jit
import jax.numpy as jnp
import jax
from operator import itemgetter
jax.config.update('jax_platform_name', 'cpu')
# the forward and backpropogation 
# are from https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180
# but added jit for faster speed in the calculation
@jit
def affine_forward(x, w, b):
    """
    Forward pass of an affine layer
    :param x: input of dimension (D, )
    :param w: weights matrix of dimension (D, M)
    :param b: biais vector of dimension (M, )
    :return output of dimension (M, ), and cache needed for backprop
    """
    out = jnp.dot(x, w) + b
    cache = (x, w)
    return out, cache

@jit
def affine_backward(dout, cache):
    """
    Backward pass for an affine layer.
    :param dout: Upstream Jacobian, of shape (O, M)
    :param cache: Tuple of:
      - x: Input data, of shape (D, )
      - w: Weights, of shape (D, M)
    :return the jacobian matrix containing derivatives of the O neural network outputs with respect to
            this layer's inputs, evaluated at x, of shape (O, D)
    """
    x, w = cache
    dx = jnp.dot(dout, w.T)
    return dx

@jit
def relu_forward(x):
    """ Forward ReLU
    """
    out = jnp.maximum(jnp.zeros(x.shape).astype(jnp.float32), x)
    cache = x
    return out, cache

@jit
def relu_backward(dout, cache):
    """
    Backward pass of ReLU
    :param dout: Upstream Jacobian
    :param cache: the cached input for this layer
    :return: the jacobian matrix containing derivatives of the O neural network outputs with respect to
             this layer's inputs, evaluated at x.
    """
    x = cache
    dx = dout * jnp.where(x > 0, jnp.ones(x.shape).astype(jnp.float32), jnp.zeros(x.shape).astype(jnp.float32))
    return dx

# @jit
def doback(dout_cache, cache_as, cache_rs, nLayers):
    last = cache_as[nLayers - 2]
    for i in range(cache_rs[0].shape[0]):
        dout = np.eye(32,32).astype(np.float32)
        for j in range(nLayers-2):
            dout   = affine_backward(dout, cache_as[j])
            dout   =   relu_backward(dout, cache_rs[j+1][i])
        dout_cache[i] = affine_backward(dout, last)   
    return dout_cache


def predict(inputs, arrModel, cal_jac=False):
    nLayers = int(len(arrModel) / 2)
    if nLayers < 2:
        raise IOError('At least one hidden layer is required')
    r = inputs.astype(jnp.float32)
    cache_as = []
    cache_rs = []

    for i in range(nLayers - 1):
        w, b = arrModel[i*2], arrModel[i*2 + 1]
        a, cache_a = affine_forward(r, w, b) 
        r, cache_r = relu_forward(a)
        cache_as.append(cache_a)
        cache_rs.append(cache_r)

    cache_as = cache_as[::-1]
    cache_rs = cache_rs[::-1]

    outs  = []
    douts = []

    for i in range(len(arrModel[-1])):
        w, b = arrModel[-2][:, [i]], arrModel[-1][i]
        out, cache_out = affine_forward(r, w, b) 
        outs.append(out)
        if cal_jac:
            dout = jnp.ones_like(out)
            dout = affine_backward(dout, cache_out)
            dout =   relu_backward(dout, cache_rs[0])
            for j in range(nLayers-2):
                dout = affine_backward(dout, cache_as[j])
                dout =   relu_backward(dout, cache_rs[j+1])
            dout = affine_backward(dout, cache_as[nLayers - 2])
            douts.append(dout)

    outs  = jnp.array(outs)
    douts = jnp.array(douts)
    
    if cal_jac:
        return outs, douts
    else:
        return outs

def round_predict(inputs, arrModel, decimals=2, cal_jac = False):
    ar = np.around(inputs, decimals=decimals)
    ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
    _, unique_row_indices = np.unique(ar_row_view, return_index=True)
    unique_p = ar[unique_row_indices]
    unique_p_row_view = ar_row_view[unique_row_indices]
    # ret = rho_tnns[i].predict(unique_p, cal_jac=True)[0][0]#.reshape(aotb.shape + (4, ))
    
    ret = predict(unique_p.astype(np.float32), arrModel, cal_jac=cal_jac)

    rho_ind = tuple(range(len(unique_p)))
    unique_p_row_view = list(map(tuple, unique_p_row_view))
    unique_dict = dict(zip(unique_p_row_view, rho_ind))
    key = list(map(tuple, ar_row_view))
    if len(key) ==1:
        inds = [itemgetter(*key)(unique_dict), ]   
    else:
        inds = list(itemgetter(*key)(unique_dict))
    if cal_jac:
        return ret[0][:, inds], ret[1][:, inds]
    else:
        return ret[:, inds]