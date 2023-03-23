#%% sparse代码：仅class的定义不同

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# class SVM: # class的定义跟非sparse的不同
#     def __init__(self,lam=0.1,delta=0.001):
#         # 参数
#         self.lam = lam
#         self.delta = delta
        
#     def train_load(self,A,b):
#         # 样本数据集
#         self.A = A
#         self.b = b
#         self.dim = A.shape[1]
    
#     def f(self,x,y):
#         ans = 0
#         for i in range(self.A.shape[0]):      
            
#             t = 1-self.b[i]*(self.A[i,:]@x+y)
            
#             if t<=0:
#                 fi = 0
#             elif t<=self.delta:
#                 fi = t**2/(2*self.delta)
#             else:
#                 fi = t-self.delta/2
#             ans += fi
#         ans = ans + np.linalg.norm(x)**2*self.lam/2    
#         return ans[0] #增加索引0
    
#     def grad(self,x,y):
#         gradx = np.zeros((self.dim,1))
#         grady = 0 
        
#         for i in range(self.A.shape[0]):
            
#             t = 1-self.b[i]*((self.A[i,:]@x)[0]+y) #增加索引[0]
#             if t<=0:
#                 fi = 0
#             elif t<=self.delta:
#                 fi = t/self.delta
#             else:
#                 fi = 1
            
#             fi = float(fi)
#             gradx += -fi*self.b[i]*self.A[i,:].reshape(-1,1)
#             grady += -fi*self.b[i]
            
#         #gradx = gradx.reshape(-1)
#         #gradx = np.ndarray.tolist(gradx)[0]
#         gradx = np.array(gradx).reshape(-1)
#         gradx = gradx+self.lam*x
    
#         return np.append(gradx,grady)




#%% 非sparse的class定义

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self,lam=0.1,delta=0.001):
        # 参数
        self.lam = lam
        self.delta = delta
        
    def train_load(self,A,b):
        # 样本数据集
        self.A = A
        self.b = b
        self.dim = A.shape[1]
    
    def f(self,x,y):
        ans = 0
        for i in range(self.A.shape[0]):      
            
            t = 1-self.b[i]*(self.A[i,:]@x+y)
            
            if t<=0:
                fi = 0
            elif t<=self.delta:
                fi = t**2/(2*self.delta)
            else:
                fi = t-self.delta/2
            ans += fi
        ans = ans + np.linalg.norm(x)**2*self.lam/2    
        return ans
    
    def grad(self,x,y):
        gradx = np.zeros((self.dim,1))
        grady = 0 
        
        for i in range(self.A.shape[0]):
            
            t = 1-self.b[i]*(self.A[i,:]@x+y)
            if t<=0:
                fi = 0
            elif t<=self.delta:
                fi = t/self.delta
            else:
                fi = 1
            
            fi = float(fi)
            gradx += -fi*self.b[i]*self.A[i,:].reshape(-1,1)
            grady += -fi*self.b[i]
            
        gradx = gradx.reshape(-1)
        gradx = gradx+self.lam*x
    
        return np.append(gradx,grady)
    
    
#%%

def line_search(model,x,d,sig,gam):
    a = 1
    while (model.f((x+a*d)[:-1], (x+a*d)[-1]) >  \
           model.f(x[:-1], x[-1]) + gam*a*model.grad(x[:-1],x[-1]) @ d ):
        a *= sig
    return a

def BFGS(model,x0=None,H0=None,eps=1e-4,max_it=1000,sig=0.5,gam=0.5): 
    # x为x,y合并
    if x0 is None:
        x = np.zeros(model.dim+1)
    else:
        x = x0
        
    if H0 is None:
        H = np.eye(model.dim+1)
    else:
        H = H0
    
    
    xy_k_list = []
    gradxy_k_list = []
    
    i = 0
    grad = model.grad(x[:-1],x[-1])
    while np.linalg.norm(grad) > eps and i<max_it:#???
        
        d = -H @ grad.T
        a = line_search(model,x,d,sig,gam)
        #line_search(f, grad, sig, gam) #??
        
        x_new = x + a*d
        s = x_new - x
        y = model.grad(x_new[:-1],x_new[-1]) - grad
        if s@y > 1e-14:
            term1 = (s-H@y).reshape(-1,1) @ s.reshape(1,-1) + s.reshape(-1,1) @ (s-H@y).reshape(1,-1)
            term1 /= s@y
            term2 = (s-H@y)@y / (s@y)**2 * (s.reshape(-1,1)@s.reshape(1,-1))
            H = H + term1 - term2
        
        x = x_new
        grad = model.grad(x[:-1],x[-1])
        i += 1
        
        # 记录过程数据
        if i%5==0:
            print('i=',i,'grad:',grad)       
        xy_k_list.append(x)
        gradxy_k_list.append(grad)
        
        
    return xy_k_list, gradxy_k_list

def acc(A_test,b_test,para):
    A_test = np.concatenate([A_test, np.ones(shape=(A_test.shape[0], 1))], axis= 1)
    b_pred = A_test@para
    b_pred[b_pred>=0] = 1
    b_pred[b_pred<0] = -1
    acc = sum(b_pred==b_test)/A_test.shape[0]
    return acc
#%%
from sklearn.model_selection import train_test_split
import time

def numgen(m1,m2,c1,c2,var1,var2):
    np.random.seed(0)
    eps1 = np.random.normal(loc=0,scale=var1,size=(2,m1))
    eps2 = np.random.normal(loc=0,scale=var2,size=(2,m2))
    m = m1+m2
    
    A = np.zeros((2,m))
    A[:,0:m1] = c1.reshape(-1,1) + eps1[:,0:m1]
    A[:,m1:m] = c2.reshape(-1,1) + eps2[:,0:m2]
    
    #前面1,后面-1
    b = np.concatenate((np.ones(m1), -np.ones(m2)))
    return A, b

m1 = 200
m2 = 188
c1 = np.array([3,4])
c2 = np.array([10,27])
var1 = 8
var2 = 4
A,b = numgen(m1,m2,c1,c2,var1,var2)
A=A.T
A_train, A_test, b_train, b_test = train_test_split(A, b, train_size=0.7, test_size=0.3, random_state=1)

plt.scatter(A[:200,0],A[:200,1],s=10,c='g',marker='o',alpha=0.7)
plt.scatter(A[200:,0],A[200:,1],s=10,c='r',marker='o',alpha=0.7)


# fit the model
model = SVM()
model.train_load(A,b)

start =time.perf_counter()
[xy_k_list,gradxy_k_list] = BFGS(model,x0=np.array([0,0,0]))
end = time.perf_counter()
time0 = end-start

para = xy_k_list[-1]
left = np.min(A)
right = np.max(A)
plot_x = np.linspace(left, right)
plot_y = -(para[-1]+para[0]*plot_x)/para[1]
plt.plot(plot_x,plot_y)

plt.figure(figsize = (20, 15))
ax1 = plt.subplot(1,2,1)
gradnorm = np.linalg.norm(gradxy_k_list,axis=1)
xx = list(range(1,gradnorm.size+1))
ax1.plot(xx,gradnorm)

cost = []
fstar = model.f(xy_k_list[-1][:-1],xy_k_list[-1][-1])
for i in range(len(xy_k_list)):
    cost.append((model.f(xy_k_list[i][:-1],xy_k_list[i][-1])-fstar)/max(1,np.abs(fstar)))
ax2 = plt.subplot(1,2,2)
ax2.plot(xx,cost)


#%%
# import numpy as np
# import scipy.io

# data_file = 'news20'

# mat_contents = scipy.io.loadmat(f'datasets/{data_file}/{data_file}_train.mat')
# A = mat_contents['A']

# mat_contents = scipy.io.loadmat(f'datasets/{data_file}/{data_file}_train_label.mat')
# b = mat_contents['b'].reshape(-1)


# # A = A.toarray()

# model =SVM()
# model.train_load(A,b)

# BFGS(model)

