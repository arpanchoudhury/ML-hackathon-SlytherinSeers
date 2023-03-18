import os
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import krr_utils
import random

#================= SMILES to fingerprint ================= 
df = pd.read_csv('dataset_train.csv', header=None)
props = df.iloc[:,1:].to_numpy()

def smile2fing(string):
    smile = string
    mol = Chem.MolFromSmiles(smile)
    #fingerprint_rdk = RDKFingerprint(mol)
    fingerprint_rdk = RDKFingerprint(mol)
    #print(">>> RDK Fingerprint = ", fingerprint_rdk)
    fingerprint_rdk_np = np.array(fingerprint_rdk)
    #print(">>> RDK Fingerprint in numpy = ", fingerprint_rdk_np)
    #print(">>> RDK Fingerprint in numpy shape = ", fingerprint_rdk_np.shape)
    return list(fingerprint_rdk_np)

rep = []
for i in range(df.shape[0]):
    x = smile2fing(df.iloc[i,0])
    rep.append(x)
rep = np.array(rep)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=1000)


np.savetxt('X_train.dat', X_train)
"""np.savetxt('X_test.dat', X_test)
np.savetxt('y_train.dat', Y_train)
np.savetxt('y_test.dat', Y_test)"""


indices = list(i_mol for i_mol in range(X_train.shape[0]))
indices_q = list(i_mol for i_mol in range(X_test.shape[0]))

#================= Kernel specific inputs ================= 
kernel = 'laplacian'
load_K = False
file_kernel = 'kernel.npy'
lamd = 1e-4
opt_sigma = krr_utils.single_kernel_sigma(500, X_train, indices, kernel, 'max')
with open('opt_sigma_'+kernel+'Kernel_fingerPrint.dat', 'w+') as f:
    f.write(str(opt_sigma))


with open('MAE_'+kernel+'Kernel_fingerPrint.dat', 'w+') as f:
    f.write('\n')


#================= Training & prediction ================= 
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
    with open('MAE_'+kernel+'Kernel_fingerPrint.dat', 'a') as f:
        f.write(str(N_train)+','+str(MAE)+'\n')

