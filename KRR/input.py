import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import krr_utils
import random


'''df = pd.read_csv('sorted_coulombMatrix.csv', header=None)
X = df.iloc[:,:].to_numpy()'''

"""X = np.loadtxt('rep.dat')
print(X.shape)
Y = np.loadtxt('props_old.dat')
print(Y.shape)
"""
X_train = np.load('rep_train.npy')
X_test = np.load('rep_test.npy')
Y_train = np.load('props_train.npy')
Y_test = np.load('props_test.npy')

"""rand_idx = list(random.sample(range(X.shape[0]), X.shape[0]))
X = X[rand_idx,:]
Y = Y[rand_idx,:]"""

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=1000)

"""np.savetxt('X_test.dat', X_test)
np.savetxt('X_train.dat', X_train)
np.savetxt('y_test.dat', Y_test)
np.savetxt('y_train.dat', Y_train)
"""

indices = list(i_mol for i_mol in range(X_train.shape[0]))
indices_q = list(i_mol for i_mol in range(X_test.shape[0]))

#--------------- Kernel specific inputs ---------------------------
kernel = 'gaussian'
load_K = False
file_kernel = 'kernel.npy'
lamd = 1e-4
opt_sigma = krr_utils.single_kernel_sigma(500, X_train, indices, kernel, 'max')
with open('opt_sigma_'+kernel+'Kernel_fingerPrint2.dat', 'w+') as f:
    f.write(str(opt_sigma))


with open('MAE_'+kernel+'Kernel_fingerPrint2.dat', 'w+') as f:
    f.write('\n')


#------------------- Training & prediction -------------------------
#for N_train in [1,10,100,1000,4000]:   
for N_train in [4000]:
    K, P = krr_utils.prepare_trainingdata(kernel,N_train,load_K,file_kernel,indices,lamd,X_train,Y_train,opt_sigma) 

    print('solving matrix equation...')
    alpha = krr_utils.linalg_solve(K,P)
    
    np.savetxt(kernel+'_Kernel_alpha.dat', alpha)

    print('predicting...')
    out_of_sample_mae = []
    y_pred_list = []
    for iquery in range(X_test.shape[0]):
        y_pred, _ = krr_utils.predict(kernel,X_train,X_test,alpha,indices,indices_q,iquery,opt_sigma)
        y_act = Y_test[indices_q[iquery],:]

        phi = np.abs(y_pred - y_act)
        #phi = (y_pred - y_act)**2
        out_of_sample_mae.append(phi)
        y_pred_list.append(y_pred)


    MAE = np.mean(out_of_sample_mae, axis=0)
    y_pred_list = np.array(y_pred_list)
    np.savetxt('y_pred_list.dat', y_pred_list)
    #print(N_train)
    #print(MAE)'''
    with open('MAE_'+kernel+'Kernel_fingerPrint2.dat', 'a') as f:
        f.write(str(N_train)+','+str(MAE)+'\n')

