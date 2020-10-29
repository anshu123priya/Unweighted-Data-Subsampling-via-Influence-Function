# -*- coding: utf-8 -*-
"""Do inverse hessian-vector-product.
"""
from optimize.optimize import fmin_ncg
import pdb
import time
import numpy as np

from scipy import sparse


def grad_logloss_theta_lr(label,ypred,x,weight_ar=None,C=0.03,has_l2=True,scale_factor=1.0):
    """Return d l_i / d_theta = d l_i / d_ypred * d y_pred / d theta
    Return:
        grad_logloss_theta: gradient on the theta, shape: [n,]
    """
    # Complex approach
    # grad_logloss_ypred = (1 - label) / (1 - ypred + 1e-10) - label / (ypred + 1e-10)
    # grad_ypred_theta = ypred * (1 - ypred) * x
    # grad_logloss_theta = grad_logloss_ypred * grad_ypred_theta

    # if there is only one sample in this batch
    if not isinstance(label,np.ndarray) or not isinstance(ypred,np.ndarray):
        label = np.array(label).flatten()
        ypred = np.array(ypred).flatten()

    if weight_ar is not None:
        # clip the feature index which is not seen in training set
        weight_len = weight_ar.shape[0]
        if x.shape[1] > weight_len:
            x = x[:, :weight_len]

    if has_l2:
        grad_logloss_theta = weight_ar + C * x.T.dot(ypred-label)
    else:
        grad_logloss_theta = C * x.T.dot(ypred-label)

    return scale_factor * grad_logloss_theta


def hessian_logloss_theta_lr(label,ypred,x,C=0.03,has_l2=True,scale_factor=1.0):
    """Get hessian matrix of logloss on theta.
    """
    assert C >= 0.0
    # if there is only one sample in this batch
    if not isinstance(label,np.ndarray) or not isinstance(ypred,np.ndarray):
        label = np.array(label).flatten()
        ypred = np.array(ypred).flatten()
        
    h = x.multiply((ypred * (1 - ypred)).reshape(-1,1))

    if isinstance(x, sparse.coo_matrix):
        h = h.tocsr()
        hessian = C * h.T.dot(x)
    else:
        hessian = C * np.matmul(h.T,x)

    if has_l2:
        diag_idx = np.arange(hessian.shape[0])
        hessian[diag_idx,diag_idx] += 1.0

    return scale_factor * hessian

def hessian_vector_product_lr(label,ypred,x,v,C=0.03,has_l2=True,scale_factor=1.0):
    """Get implicit hessian-vector-products without explicitly building the hessian.
    H*v = v + C *X^T*(D*(X*v))
    """
    xv = x.dot(v)
    D = ypred * (1 - ypred)
    dxv = xv * D
    if has_l2:
        hvp = C * x.T.dot(dxv) +  v
    else:
        hvp = C * x.T.dot(dxv)

    return scale_factor * hvp


def inverse_hvp_lr_newtonCG(x_train,y_train,y_pred,v,C=0.01,hessian_free=True,tol=1e-5,has_l2=True,M=None,scale_factor=1.0):
    """Get inverse hessian-vector-products H^-1 * v
    Return:
        H^-1 * v: shape [None,]
    """
    if not hessian_free:
        hessian_matrix = hessian_logloss_theta_lr(y_train,y_pred,x_train,C,has_l2,scale_factor)

    # build functions for newton-cg optimization
    def fmin_loss_fn(x):
        """Objective function for newton-cg.
        H^-1 * v = argmin_t {0.5 * t^T * H * t - v^T * t}
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor) # [n,]
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]
        obj = 0.5 * np.dot(hessian_vec_val,x) - \
                    np.dot(x, v)

        return obj

    def fmin_grad_fn(x):
        """Gradient of the objective function w.r.t t:
        grad(obj) = H * t - v
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor)
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]

        grad = hessian_vec_val - v

        return grad

    def get_fmin_hvp(x,p):
        # get H * p
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,p,C,has_l2,scale_factor)
        else:
            hessian_vec_val = np.dot(p,hessian_matrix)

        return hessian_vec_val

    def get_cg_callback(verbose):
        def fmin_loss_split(x):
            if hessian_free:
                hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor)
            else:
                hessian_vec_val = np.dot(x,hessian_matrix)

            loss_1 = 0.5 * np.dot(hessian_vec_val,x)
            loss_2 = - np.dot(v, x)
            return loss_1, loss_2

        def cg_callback(x):
            if verbose:
                print("Function value:", fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print("Split function value: {}, {}".format(quad, lin))
        return cg_callback

    start_time = time.time()
    cg_callback = get_cg_callback(verbose=True)
    fmin_results = fmin_ncg(f=fmin_loss_fn,
                           x0=v,
                           fprime=fmin_grad_fn,
                           fhess_p=get_fmin_hvp,
                           callback=cg_callback,
                           avextol=tol,
                           maxiter=100,
                           preconditioner=M)
    print("Inverse HVP took {:.1f} sec".format(time.time() - start_time))
    return fmin_results