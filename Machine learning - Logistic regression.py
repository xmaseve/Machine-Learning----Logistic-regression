# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 11:24:24 2016

@author: YI
"""

import numpy as np
from scipy import optimize

#read data
data = np.loadtxt('ex2data1.txt', delimiter=',')
x = data[:,0:2]
y = data[:,2]
m = len(y)
X = np.ones((m,3))
X[:,1:] = x
y = y.reshape((m,1))
n = len(X[0])

#sigmoid function
def sigmoid(x):
    g = 1./(1+np.exp(-x))
    return g

#loss function
theta = np.zeros((n,1))
lamda = 0.1


def loss(theta,x,y,lamda):
    z = x.dot(theta)
    J = ((-y).T.dot(np.log(sigmoid(z)))-\
    (1-y).T.dot(np.log(1-sigmoid(z))))/m
    R = lamda*np.sum(theta[1:]**2)/(2*m)  #no intercept term
    L = J + R
    return J,R,L
    
loss(theta,X,y,lamda)

#gradient descent
num_iters = 2000
alpha = 0.01

def gradient_descent(theta,x,y,alpha=0.01,num_iters=2000):
    
    for i in xrange(num_iters):
        dtheta0 = np.sum(sigmoid(x.dot(theta))-y)/m
        theta[0][0] += -alpha * dtheta0
        
        for u in xrange(1,n):
            dJ = np.sum((sigmoid(x.dot(theta))-y)*x[:,u])/m
            dR = lamda*theta[u][0]/m
            theta[u][0] += -alpha * (dJ + dR)
            
    return theta
    
gradient_descent(theta,X,y)

##################
#advanced optimization
def optimization(theta,x,y,lamda):
    result = optimize.fmin(loss, x0=theta, args=(x, y, lamda), maxiter=400,\
    full_output=True)
    return result[0], result[1]

optimization(theta,X,y,lamda)












            
        
    
    

    