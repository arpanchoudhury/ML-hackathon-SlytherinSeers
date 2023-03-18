import numpy as np
import krr_utils.kernels as kernels



def single_kernel_sigma(N_sample, X, indices, kernel, typ):
    """
    Calculates the kernel width parameter 'sigma' via 

        sigma = D_ij / log(2)			; for Laplacian kernel
        sigma = D_ij / sqrt( 2 * log(2) )      	; for Gaussian kernel

    where 'D_ij' is the largest distance between i-th and j-th input vector.

    N_sample: 'sigma' is calculated over 'N_sample' data from training set.
    X: input array
    indices: index array required for random selection of 'N_sample'.
    kernel: kernel specification ['laplacian','gaussian'].
    typ:  type of distance measure ['max','median'].

    return_type: scalar
    """

    D = []
    for i in range(0,N_sample-1):
        for j in range(i+1,N_sample):
            Dij = np.sum(np.abs(X[indices[i]] - X[indices[j]]))
            D.append(Dij)
            
    D = np.array(D,dtype=float)

    Dmax = np.max(D)
    Dmedian = np.median(D)

    if kernel == 'laplacian':
        if typ == 'max':
            opt_sigma = Dmax/(np.log(2.0))
        elif typ == 'median':
            opt_sigma = Dmedian/(np.log(2.0))
    elif kernel == 'gaussian':
        if typ == 'max':
            opt_sigma = Dmax/np.sqrt(2.0*(np.log(2.0)))
        elif typ == 'median':
            opt_sigma = Dmedian/np.sqrt(2.0*(np.log(2.0)))

    return opt_sigma


def prepare_trainingdata(kernel,N_train,load_K,file_kernel,indices,lamd,X,y,opt_sigma):
    """
    Generates full kernel matrix 'K' and property matrix 'P'
    
    kernel: kernel specification ['laplacian','gaussian'].
    N_train: size of the training set.
    load_K: whether to load 'K' from an existing file [bool].
    file_kernel: load 'K' from this file.
    indices: index array required for random selection of training set.
    lamd: regulerization strength lambda.
    X: input array.
    y: output array.
    opt_sigma: optimized value of the kernel width sigma.

    return_type: numpy array 'K[N_train,N_train]', numpy array 'P[N_train,N_prop]'.
    where 'N_prop' is the total number of output.
    """

    if load_K:
        K = np.load(file_kernel)
    else:
        K = np.zeros([N_train,N_train], dtype=float)
        print('Calculating kernel matrix'+'\n')

        for itrain in range(N_train):
            K[itrain,itrain] = 1.0 + lamd
            if np.mod(itrain,10) == 0:
                print(itrain, 'rows calculated,', N_train-itrain, 'remaining')
            for jtrain in range(itrain+1,N_train):
                Xt = X[indices[itrain]]
                Xq = X[indices[jtrain]]
                Yt = np.zeros([1,len(Xt)],dtype=float)
                Yq = np.zeros([1,len(Xq)],dtype=float)
                Yt[0] = Xt
                Yq[0] = Xq

                if kernel == 'laplacian':
                    tmp = kernels.laplacian_kernel(Yq,Yt, sigma=opt_sigma)
                elif kernel == 'gaussian':
                    tmp = kernels.gaussian_kernel(Yq,Yt, sigma=opt_sigma)

                K[itrain,jtrain] = tmp[0,0]
                K[jtrain,itrain] = K[itrain,jtrain]

        np.save(file_kernel, K)
    #print('Kernel matrix done.')

    N_prop = len(y[0])
    P = np.zeros([N_train,N_prop],dtype=float)

    for iprop in range(N_prop):
        for itrain in range(N_train):
            P[itrain,iprop] = y[indices[itrain],iprop]
    #print('property matrix size:', P.shape)

    return K, P

