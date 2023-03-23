# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 19:13:08 2022

@author: Administrator
"""

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split

#%% log
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def f_log(xy,A,b,lam):
    x = np.array(xy[:-1])
    y = np.array(xy[-1])
    b = np.array(b).reshape(-1)
    z = (A@x+y)*b
    m = b.size
    J = -1 / m * np.sum(np.log(sigmoid(z)))+np.linalg.norm(x)**2*lam/2 
    return J
 
def grad_log(xy,A,b,lam):
    x = np.array(xy[:-1])
    y = np.array(xy[-1])
    b = np.array(b).reshape(-1)
    z = (A@x+y)*b
    m = b.size
    
    gradx = np.zeros((A.shape[1],1))
    grady = 0 
    
    for i in range(m):
        gradx += -np.exp(-z[i])*sigmoid(z[i])*b[i]*A[i,:].reshape(-1,1)
        grady += -np.exp(-z[i])*sigmoid(z[i])*b[i]

    grady = grady/m
    gradx = np.ndarray.tolist(gradx)[0]
    gradx = np.array(gradx)
    gradx = gradx/m+lam*x

    return np.append(gradx,grady)

def AGMbacktracking(opt,A,b,lam,f,grad,gamma,sigma,t0,t1,initial_xy):
    # set maximum number of iteration
    max_iter=1000
    # set stop criteria
    stop_criteria = 1E-5
    # set x to initial point
    xy_k = np.array(initial_xy)
    xy0 = xy_k.copy()
    # create lists to store result
    s_list = []
    xy_k_list = []
    gradxy_k_list = []
    # iterate until maximum iteration is reached 
    for i in tqdm(range(max_iter)):
        if opt == 'agd':
            beta = (t0-1)/t1
            # calcualte gradient value
            xy_kk = xy_k+beta*(xy_k-xy0)
            xy0 = xy_k
            gradxy = grad(xy_kk,A,b,lam)
            # set s to initial s
            s = 1
            # check stop criteria
            gradsqr = np.linalg.norm(gradxy)**2
            if np.sqrt(gradsqr) > stop_criteria:
                # next point
                xy_k_plus_one = xy_kk-s*gradxy
                # check if the sufficient decrease condition is met
                iters = 0
                while (f(xy_k_plus_one,A,b,lam) - f(xy_kk,A,b,lam)) >= -(gamma*s*gradsqr):
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
            else:
                break
        else:
            gradxy = grad(xy_k,A,b,lam)
            # set s to initial s
            s = 1
            # check stop criteria
            gradsqr = np.linalg.norm(gradxy)**2
            if np.sqrt(gradsqr) > stop_criteria:
                # next point
                xy_k_plus_one = xy_k-s*gradxy
                # check if the sufficient decrease condition is met
                iters = 0
                while (f(xy_k_plus_one,A,b,lam) - f(xy_k,A,b,lam)) >= -(gamma*s*gradsqr):
                    # update step length
                    s=sigma*s
                    iters += 1
                    # re-calculate next point
                    xy_k_plus_one = xy_k-s*gradxy
                    if iters>50:
                        break
                xy_k=xy_k_plus_one
            # if the stop criteria is satisfied
            else:
                break
        # put result in lists
        gradxy_k_list.append(gradxy)
        s_list.append(s)
        xy_k_list.append(xy_k)
    return xy_k_list,gradxy_k_list

def acc(A_test,b_test,para):
    b_pred = sigmoid(A_test@para[:-1]+para[-1])
    b_pred[b_pred>=0.5] = 1
    b_pred[b_pred<0.5] = -1
    b_pred = b_pred.reshape(-1,1)
    boole = b_pred==b_test
    acc = np.sum(boole!=0)/A_test.shape[0]
    return acc

#%%
mat_contents = scipy.io.loadmat('datasets/gisette/gisette_train.mat')
A = mat_contents['A']
#A_train = A

mat_contents = scipy.io.loadmat('datasets/gisette/gisette_train_label.mat')
b = mat_contents['b']

A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.8, test_size=0.2, random_state=0)

# mat_contents = scipy.io.loadmat('datasets/gisette/gisette_test.mat')
# A_test = mat_contents['A']

# mat_contents = scipy.io.loadmat('datasets/gisette/gisette_test_label.mat')
# b_test = mat_contents['b']

lam = 1/A_train.shape[0]
delta = 0.01
gamma = 0.1
sigma = 0.5
t0 = 1
t1 = 1
initial_xy = [0]*(A.shape[1]+1)

[xy_k_list,gradxy_k_list] = AGMbacktracking('agd',A_train,b_train,lam,f_log,grad_log,gamma,sigma,t0,t1,initial_xy)
para = xy_k_list[-1]
acc = acc(A_test,b_test,para)

plt.figure(figsize = (20, 15))
ax1 = plt.subplot(1,2,1)
gradnorm = np.linalg.norm(gradxy_k_list,axis=1)
xx = list(range(1,gradnorm.size+1))
ax1.plot(xx,gradnorm)

cost = []
fstar = f_log(xy_k_list[-1],A_train,b_train,lam)
for i in range(len(xy_k_list)):
    cost.append(np.abs((f_log(xy_k_list[i],A_train,b_train,lam)-fstar))/max(1,np.abs(fstar)))
ax2 = plt.subplot(1,2,2)
ax2.plot(xx,cost)