import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split

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

def grad_SVM(xy,A,b,lam,delta):
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
    
    # grad = np.zeros(3)
    gradx = np.zeros((A.shape[1],1))
    grady = 0 
    
    for i in range(A.shape[0]):
        
        t = 1-b[i]*(A[i,:]@x+y)
        if t<=0:
            fi = 0
        elif t<=delta:
            fi = t/delta
        else:
            fi = 1
        fi = float(fi)
        gradx += -fi*b[i]*A[i,:].reshape(-1,1)
        grady += -fi*b[i]
        
    gradx = np.ndarray.tolist(gradx)[0]
    gradx = np.array(gradx)
    gradx = gradx+lam*x

    return np.append(gradx,grady)

def AGMbacktracking(A,b,lam,delta,f,grad,gamma,sigma,t0,t1,initial_xy):
    # set maximum number of iteration
    max_iter=1000
    # set stop criteria
    stop_criteria = 1E-5
    # set x to initial point
    xy_k = np.array(initial_xy)
    xy0 = xy_k.copy()
    # create lists to store result
    s_list = []
    #x_k_list = []
    #y_k_list = []
    xy_k_list = []
    gradxy_k_list = []
    # iterate until maximum iteration is reached 
    for i in tqdm(range(max_iter)):
        beta = (t0-1)/t1
        # calcualte gradient value
        xy_kk = xy_k+beta*(xy_k-xy0)
        xy0 = xy_k
        #y_kk = y_k+beta*(y_k-y0)
        #y0 = y_k
        #xy_kk = 
        gradxy = grad_SVM(xy_kk,A,b,lam,delta)
        # set s to initial s
        s = 1
        # check stop criteria
        gradsqr = np.linalg.norm(gradxy)**2
        if np.sqrt(gradsqr) > stop_criteria:
            # next point
            xy_k_plus_one = xy_kk-s*gradxy
            # check if the sufficient decrease condition is met
            iters = 0
            while (f(xy_k_plus_one,A,b,lam,delta) - f(xy_kk,A,b,lam,delta)) >= -(gamma*s*gradsqr):
                # update step length
                s=sigma*s
                iters += 1
                # re-calculate next point
                xy_k_plus_one = xy_kk-s*gradxy
                if iters>50:
                    break
            # update x
            xy_k=xy_k_plus_one
            gu = t1
            t1 = 0.5*(1+np.sqrt(1+4*t0*t0))
            t0 = gu
            # put result in lists
            gradxy_k_list.append(gradxy)
            s_list.append(s)
            xy_k_list.append(xy_k)
        # if the stop criteria is satisfied
        else:
            break
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
mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_train.mat')
A = mat_contents['A']
A_train = A

mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_train_label.mat')
b_train = mat_contents['b']

#A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.7, test_size=0.3, random_state=1)

mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_test.mat')
A_test = mat_contents['A']

mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_test_label.mat')
b_test = mat_contents['b']

lam = 0.1
delta = 1
gamma = 0.1
sigma = 0.5
t0 = 1
t1 = 1
initial_xy = [0]*(A.shape[1]+1)


[xy_k_list,gradxy_k_list] = AGMbacktracking(A_train,b_train,lam,delta,f_SVM,grad_SVM,gamma,sigma,t0,t1,initial_xy)
para = xy_k_list[-1]
acc = acc(A_test,b_test,para)