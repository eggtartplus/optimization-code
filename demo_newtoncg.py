# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:10:39 2022

@author: Administrator
"""
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
import random
import time

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
    
    # gradx = gradx.todense().T.tolist()[0]
    # gradx = np.array(gradx)
    gradx = np.array(gradx).reshape(-1)
    gradx = gradx+lam*x
    grady = grady/m

    return np.append(gradx,grady)

def hessian_xy(xy, A, b, lam):
    x = np.array(xy[:-1])
    y = np.array(xy[-1])
    b = np.array(b).reshape(-1)
    m = b.size
    n = x.shape[0]
    
    hessian_x = np.zeros((n, n))
    hessian_xy = np.zeros((x.shape[0], 1))
    hessian_y = 0
    
    for i in range(m):
        ai = A[i,:].reshape(-1, 1)
        hessian_0 = b[i]*b[i] * np.exp(b[i]*(ai.T@x+y))/ np.square(1+np.exp(b[i]*(ai.T@x+y)))
        hessian_x = hessian_x + hessian_0[0] * np.dot(ai, ai.T)
        hessian_y = hessian_y + hessian_0[0]
        hessian_xy = hessian_xy + hessian_0[0]*ai.reshape(-1,1)
    hessian_x = hessian_x/m + lam
    hessian_xy = hessian_xy/m 
    hessian_y = hessian_y/m
    hessian = np.zeros((n+1,n+1))
    hessian[:n,:n] = hessian_x
    hessian[:n,n] = hessian_xy.reshape(-1)
    hessian[n,:n] = hessian_xy.reshape(-1)
    hessian[n,n] = hessian_y
    return hessian

def backtracking(f, df, A, b, x, m, lam, d, sigma, gamma):
    alpha = 1
    fx = f(x,A, b, lam)
    while True:
        xn=x+alpha*d
        fxn = f(xn, A, b, lam)
        dfx = df(x, A, b, lam)
        if fxn - fx <= gamma * alpha * dfx.T@d:
            break
        alpha *= sigma
    return alpha

def newtonCG(f, df, A, b, x, m, lam):
    # set maximum number of iteration
    max_iter=1000
    # set stop criteria
    stop_criteria = 1E-5
    # set x to initial point
    xy_k = np.array(initial_xy)
    # create lists to store result
    xy_k_list = []
    gradxy_k_list = []
    for i in tqdm(range(max_iter)):
        gradxy = grad_log(xy_k, A, b, lam)   #gd_x, 是之前用的把ai和x都变成n+1维，即ai新增一行为1  （对应现在的函数gd_xy,但是是把x y分开算的）
        if np.linalg.norm(gradxy) < stop_criteria:
            break
        A1 = hessian_xy(xy_k, A, b, lam) #hessian 矩阵
        normA1 = np.linalg.norm(A1)
        tol = min(1, np.power(normA1, 0.01)) * normA1
        v = np.zeros(xy_k.shape)
        r = gradxy
        r_cop = np.copy(r)
        p = -r
        for j in range(20):
            if j == 0:
                d = -gradxy
            tempory = p.T @ A1 @ p
            if tempory <= 0:
                if j != 0:
                    d = v
                    break
            sigma_j = np.square(np.linalg.norm(r_cop)) / tempory
            #sigma_j = float(sigma_j[0][0]) #no need with generated data
            v = v + float(sigma_j)*p
            r = r_cop + sigma_j*A1@p
            if np.linalg.norm(r) <= tol:
                d = v
                break
            beta = np.square(np.linalg.norm(r) / np.linalg.norm(r_cop))
            p = -r + beta * p
            r_cop = np.copy(r)       
        #alpha = backtracking(f, grad_log, A, b, xy_k, m, lam, d, 0.5, 0.1)
        xy_k = xy_k + 1/np.sqrt(i+1) * d
        xy_k_list.append(xy_k)
        gradxy_k_list.append(gradxy)
    return xy_k_list,gradxy_k_list

def acc(A_test,b_test,para):
    b_pred = A_test@para[:-1]+para[-1]
    b_pred[b_pred>=0.5] = 1
    b_pred[b_pred<0.5] = -1
    b_pred = b_pred.reshape(-1,1)
    boole = b_pred==b_test
    acc = np.sum(boole!=0)/A_test.shape[0]
    return acc

#%%
mat_contents = scipy.io.loadmat('datasets/mushrooms/mushrooms_train.mat')
A = mat_contents['A']
#A_train = A

mat_contents = scipy.io.loadmat('datasets/mushrooms/mushrooms_train_label.mat')
b = mat_contents['b']

A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.7, test_size=0.3, random_state=1)

# mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_test.mat')
# A_test = mat_contents['A']

# mat_contents = scipy.io.loadmat('datasets/rcv1/rcv1_test_label.mat')
# b_test = mat_contents['b']

lam = 1/A_train.shape[0]
delta = 0.01
gamma = 0.1
sigma = 0.5
t0 = 1
t1 = 1
initial_xy = [0]*(A.shape[1]+1)
m = A_train.shape[0]

start =time.perf_counter()
[xy_k_list,gradxy_k_list] = newtonCG(f_log, grad_log, A_train, b_train, initial_xy, m, lam)
end = time.perf_counter()
time0 = end-start
para = xy_k_list[-1]
acc = acc(A_test,b_test,para)
print(time0)
print(acc)
print(f_log(xy_k_list[-1],A_train,b_train,lam))