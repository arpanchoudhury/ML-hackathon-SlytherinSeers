import numpy as np
import krr_utils.kernels as kernels


def predict(kernel,X_train, X_query,alpha,indices_t,indices_q,iquery,opt_sigma):
    """
    Makes predictions on a test set.

    kernel: kernel specification ['laplacian','gaussian'].
    X_train: training input array.
    X_query: test input array.
    alpha: regression coefficients.
    indices_t: index array required for random selection of training set.
    indices_q: index array required for random selection of test set.
    iquery: make prediction for the iquery-th entry of the test set.
    opt_sigma: optimized value of the kernel width sigma. 

    return_type: 1D numpy array 'P_pred[N_prop]'
    """

    N_train = alpha.shape[0]
    N_prop = alpha.shape[1]

    K = np.zeros(N_train)

    for itrain in range(N_train):
        Xt = X_train[indices_t[itrain]]
        Xq = X_query[indices_q[iquery]]
        Yt = np.zeros([1,len(Xt)],dtype=float)
        Yq = np.zeros([1,len(Xq)],dtype=float)
        Yt[0] = Xt
        Yq[0] = Xq

        if kernel == 'laplacian':
            tmp = kernels.laplacian_kernel(Yq,Yt, sigma=opt_sigma)
        elif kernel == 'gaussian':
            tmp = kernels.gaussian_kernel(Yq,Yt, sigma=opt_sigma)

        K[itrain] = tmp[0,0]

    P_pred = np.zeros(N_prop)
    for i_prop in range(N_prop):
        P_pred[i_prop] = np.dot(K,alpha[:,i_prop])

    return P_pred, Xq

