# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:12:11 2022

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
import time

def numgen(m1,m2,c1,c2,var1,var2):
    np.random.seed(0)
    eps1 = np.random.normal(loc=0,scale=var1,size=(2,m1))
    eps2 = np.random.normal(loc=0,scale=var2,size=(2,m2))
    m = m1+m2
    A = np.zeros((2,m))
    #for i in range(m1):
    A[:,0:m1] = c1.reshape(-1,1) + eps1[:,0:m1]
    #for i in range(m1,m):
    A[:,m1:m] = c2.reshape(-1,1) + eps2[:,0:m2]
    #前面1,后面-1
    b = np.concatenate((np.ones(m1), -np.ones(m2)))
    return A, b

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
    gradx = np.zeros((2,1))
    grady = 0 
    
    for i in range(A.shape[0]):
        
        t = 1-b[i]*(A[i,:]@x+y)
        if t<=0:
            fi = 0
        elif t<=delta:
            fi = t/delta
        else:
            fi = 1
        gradx += -fi*b[i]*A[i,:].reshape(-1,1)
        grady += -fi*b[i]
        
    gradx = gradx.reshape(-1)
    gradx = gradx+lam*x

    return np.append(gradx,grady)

def AGMbacktracking(opt,A,b,lam,delta,f,grad,gamma,sigma,t0,t1,initial_xy):
    # set maximum number of iteration
    max_iter=1000
    # set stop criteria
    stop_criteria = 1E-4
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
                    if iters>100:
                        gradxy1 = grad_SVM(xy_k,A,b,lam,delta)
                        xy_k_plus_one = xy_k-s*gradxy1
                        break
                # update x
                xy_k=xy_k_plus_one
                gu = t1
                t1 = 0.5*(1+np.sqrt(1+4*t0*t0))
                t0 = gu
            else:
                break
        else:
            gradxy = grad(xy_k,A,b,lam,delta)
            # set s to initial s
            s = 1
            # check stop criteria
            gradsqr = np.linalg.norm(gradxy)**2
            if np.sqrt(gradsqr) > stop_criteria:
                # next point
                xy_k_plus_one = xy_k-s*gradxy
                # check if the sufficient decrease condition is met
                iters = 0
                while (f(xy_k_plus_one,A,b,lam,delta) - f(xy_k,A,b,lam,delta)) >= -(gamma*s*gradsqr):
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
        # if the stop criteria is satisfied
        
    return xy_k_list,gradxy_k_list

def acc(A_test,b_test,para):
    A_test = np.concatenate([A_test, np.ones(shape=(A_test.shape[0], 1))], axis= 1)
    b_pred = A_test@para
    b_pred[b_pred>=0] = 1
    b_pred[b_pred<0] = -1
    acc = sum(b_pred==b_test)/A_test.shape[0]
    return acc

#%%
m1 = 500
m2 = 500
c1 = np.array([5,4]).T
c2 = np.array([3,7]).T
var1 = 2
var2 = 1
[A,b] = numgen(m1,m2,c1,c2,var1,var2)
A = A.T
A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.7, test_size=0.3, random_state=1)

plt.scatter(A[:m1,0],A[:m1,1],s=10,c='g',marker='o',alpha=0.7)
plt.scatter(A[m1:,0],A[m1:,1],s=10,c='r',marker='o',alpha=0.7)


lam = 1/(m1+m2)
delta = 0.01
gamma = 0.1
sigma = 0.5
t0 = 1
t1 = 1
initial_xy = [1,2,3]

start =time.perf_counter()
[xy_k_list,gradxy_k_list] = AGMbacktracking('agd',A_train,b_train,lam,delta,f_SVM,grad_SVM,gamma,sigma,t0,t1,initial_xy)
end = time.perf_counter()
time0 = end-start
 
para = xy_k_list[-1]
left = np.min(A)
right = np.max(A)
plot_x = np.linspace(left, right)
plot_y = -(para[-1]+para[0]*plot_x)/para[1]
plt.plot(plot_x,plot_y)

acc = acc(A_test,b_test,para)

plt.figure(figsize = (20, 15))
ax1 = plt.subplot(1,2,1)
gradnorm = np.linalg.norm(gradxy_k_list,axis=1)
xx = list(range(1,gradnorm.size+1))
ax1.plot(xx,gradnorm)

cost = []
fstar = f_SVM(xy_k_list[-1],A_train,b_train,lam,delta)
for i in range(len(xy_k_list)):
    cost.append((f_SVM(xy_k_list[i],A_train,b_train,lam,delta)-fstar)/max(1,np.abs(fstar)))
ax2 = plt.subplot(1,2,2)
ax2.plot(xx,cost)

