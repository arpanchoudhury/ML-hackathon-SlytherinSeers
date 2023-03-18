import numpy as np
from scipy import linalg
from scipy.linalg import cho_factor, cho_solve


def linalg_solve(K,P,solver='cholesky'):
    """
    Calculates the regression coefficient by solving
        [K-lambda*I] * alpha = P

    K: kernel matrix.
    P: property matrix.
    solver: default='cholesky'.

    return_type: numpy array alpha[N_train,N_prop].
    """
    N_train = K.shape[0]
    N_prop = P.shape[1]

    alpha = np.zeros([N_train,N_prop])

    if solver == 'cholesky':
        Klow, low = cho_factor(K)
        for i_prop in range(N_prop):
            alpha[:,i_prop] = cho_solve((Klow, low), P[:,i_prop])
    else:
        alpha = linalg.solve(K, P)

    return alpha
