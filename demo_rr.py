import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
import random
import time

#%% svm
def f_SVM(xy,A,b,lam,delta):
    """ 
    Parameters
    ----------
    xy : (3,)
    A : (m,2)
    b : (m,)
    lam : float
    delta : float

    Returns
    -------
    ans : float

    """
    # 输入的x分为x,y
    x = np.array(xy[:-1])
    y = np.array(xy[-1])
    b = np.array(b).reshape(-1)
    
    ans = 0
    for i in range(A.shape[0]): 
        
        t = 1-b[i]*(A[i,:]@x+y)
        
        if t<=0:
            fi = 0
        elif t<=delta:
            fi = t**2/(2*delta)
        else:
            fi = t-delta/2
        ans += fi
    ans = ans + np.linalg.norm(x)**2*lam/2    
    return ans

def sgdgrad_SVM(xy,i,A,b,lam,delta):
    """
    Parameters
    ----------
    xy : (3,)
    A : (m,2)
    b : (m,)
    lam : float
    delta : float

    Returns
    -------
    grad : (3,)

    """
    x = np.array(xy[:-1])
    y = np.array(xy[-1])
    b = np.array(b).reshape(-1)
    
    t = 1-b[i]*(A[i,:]@x+y)
    if t<=0:
        fi = 0
    elif t<=delta:
        fi = t/delta
    else:
        fi = 1
    fi = float(fi)
    gradx = -fi*b[i]*A[i,:].reshape(-1,1)
    grady = -fi*b[i]
        
    gradx = gradx.todense().T.tolist()[0]
    gradx = np.array(gradx)
    gradx = gradx+lam*x

    return np.append(gradx,grady)

def rr(A,b,lam,delta,grad,t0,t1,initial_xy):
    m = A.shape[0]
    s = 1
    # set maximum number of iteration
    max_iter=10
    # set x to initial point
    xy_k = np.array(initial_xy)
    xy0 = xy_k.copy()
    # create lists to store result
    xy_k_list = []
    gradxy_k_list = []
    # iterate until maximum iteration is reached 
    for j in tqdm(range(max_iter)):
        for i in tqdm(range(m)):
            l = list(range(m))
            random.shuffle(l)
            # calcualte gradient value
            beta = (t0-1)/t1
            xy_kk = xy_k+beta*(xy_k-xy0)
            xy0 = xy_k
            gradxy = grad(xy_kk,l[i],A,b,lam,delta)
            s = s/np.sqrt(i+1)
            xy_k_plus_one = xy_kk-s*gradxy
            xy_k=xy_k_plus_one
            gu = t1
            t1 = 0.5*(1+np.sqrt(1+4*t0*t0))
            t0 = gu
            gradxy_k_list.append(gradxy)
            xy_k_list.append(xy_k)
       
    return xy_k_list,gradxy_k_list

def acc(A_test,b_test,para):
    b_pred = A_test@para[:-1]+para[-1]
    b_pred[b_pred>=0] = 1
    b_pred[b_pred<0] = -1
    b_pred = b_pred.reshape(-1,1)
    boole = b_pred==b_test
    acc = np.sum(boole!=0)/A_test.shape[0]
    return acc

#%%
mat_contents = scipy.io.loadmat('datasets/sido0/sido0_train.mat')
A = mat_contents['A']
#A_train = A

mat_contents = scipy.io.loadmat('datasets/sido0/sido0_train_label.mat')
b = mat_contents['b']

A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.2, test_size=0.12, random_state=0)

# mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_test.mat')
# A_test = mat_contents['A']

# mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_test_label.mat')
# b_test = mat_contents['b']

lam = 1/A.shape[0]
delta = 0.01
gamma = 0.1
sigma = 0.5
t0 = 1
t1 = 1
initial_xy = [0]*(A.shape[1]+1)

start =time.perf_counter()
[xy_k_list,gradxy_k_list] = rr(A_train,b_train,lam,delta,sgdgrad_SVM,t0,t1,initial_xy)
end = time.perf_counter()
time0 = end-start
para = xy_k_list[-1]
acc = acc(A_test,b_test,para)
print(time0)
print(acc)
print(f_SVM(xy_k_list[-1],A_train,b_train,lam,delta))