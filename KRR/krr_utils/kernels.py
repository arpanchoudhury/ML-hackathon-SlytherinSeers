import numpy as np
import krr_utils.fortran_kernels as fKernels


def laplacian_kernel(A, B, sigma):
    """
    Calculates the Laplacian kernel matrix K
        K_ij = exp( -|A_i - B_j|_1 / sigma )

    A, B: input array.
    sigma: kernel width
    """

    if A.shape[1] != B.shape[1]:
        raise Exception('Feature vectors length must be equal.')

    A_row = A.shape[0]
    B_row = B.shape[0]

    K = np.empty((A_row, B_row), order='F')
    fKernels.fortran_laplacian_kernel(A.T, B.T, A_row, B_row, K, sigma)	# Transposed for Fortran column-major computation

    return K


def gaussian_kernel(A, B, sigma):
    """
    Calculates the Gaussian kernel matrix K
        K_ij = exp( -|A_i - B_j|_2 ^2 / 2*sigma^2 )

    A, B: input array.
    sigma: kernel width
    """

    if A.shape[1] != B.shape[1]:
        raise Exception('Feature vectors length must be equal.')

    A_row = A.shape[0]
    B_row = B.shape[0]

    K = np.empty((A_row, B_row), order='F')
    fKernels.fortran_gaussian_kernel(A.T, B.T, A_row, B_row, K, sigma)	# Transposed for Fortran column-major computation

    return K
