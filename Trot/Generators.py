# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:42:22 2016

@author: boris
"""

import numpy as np
from math import ceil

maxval = 100000

#Uniform sampling over the open unit simplex (i.e. no 0s allowed)
def rand_marginal(n):
    x = np.random.choice(np.arange(1,maxval-1), n, replace = False)
    p = np.zeros(n)
    x[n-1] = maxval
    x = np.sort(x)
    for i in range(1,n):
        p[i] = (x[i]-x[i-1])/maxval
    p[0] = x[0]/maxval
    return p
    
    
#Same cost matrix as in Cuturi 2013 (for scale = 1), not necessarily square
def rand_costs(n,m,scale):
    X = np.random.multivariate_normal(np.zeros(ceil(n/10)),np.identity(ceil(n/10)),max(n,m))*scale
    M = np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            M[i,j] = np.linalg.norm(X[i]-X[j])
    
    return M/np.median(M)

#Squared Euclidean distance cost matrix
def euc_costs(n,scale):
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j]=(i/n - j/n)*(i/n - j/n)
    return M*scale

# Sqaured euclidean distance using real values
def real_euc_costs(n):
    # Create dummy euc cost matrix
    M = np.zeros((np.shape(n)[0],np.shape(n)[0]))
    
    # Normalise range
    n = ((n - np.min(n)) /  (np.max(n) - np.min(n)))
    for i,x in enumerate(n):
        for j,y in enumerate(n):
            M[i,j] = (x - y)**2
    return M*len(n)
