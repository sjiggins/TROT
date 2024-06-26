# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:38:13 2016

@author: boris
"""


import numpy as np
import math
from copy import deepcopy
from Projections import check, Sinkhorn
from Distances import inner_Frobenius


#Tsallis generalization of the exponential
def q_exp (q,u):
    return np.power(1+(1-q)*u,1/(1-q))

#Tsallis generalization of the logarithm
def q_log (q,u):
    return (np.power(u,1-q) - 1)/(1-q)

#Objective function of the TROT minimization problem
def q_obj(q,P,M,l):
    P_q = np.power(P,q)
    return inner_Frobenius(P,M) - np.sum(P_q - P)/((1-q)*l)

#Gradient of the objective function in q-space
def q_grad(q,P,M,l):
    return M + (q*np.power(P,q-1) - 1)/(l*(1-q))

def sign(x):
    if (x>0):
        return 1 
    else:
        return -1    
    
#A wrapper for the three TROT-solving algorithms, depending on the value of q
#q must be positive
def TROT(q,M,r,c,l,precision):
    
    assert (q >= 0),"Invalid parameter q: q must be strictly positive"

    if np.isclose(q,1):
        #Add multipliers to rescale A and avoid dividing by zero
        A = deepcopy(l*M)
        A = A-np.amin(A,axis = 0)
        A = (A.T-np.amin(A,axis = 1)).T
        
        return Sinkhorn(np.exp(-A),r,c,precision)
   
    elif q<1:
        
        #return third_order_sinkhorn(q,M,r,c,l,precision)[0]
        return second_order_sinkhorn(q,M,r,c,l,precision)[0]
        #return first_order_sinkhorn(q,M,r,c,l,precision)[0]

    else:
        return KL_proj_descent(q,M,r,c,l,precision, 500, rate = 1, rate_type = "square_summable")[0]
        


#A TROT optimizer using first order approximations (less efficient than second order)
def first_order_sinkhorn(q,M,r,c,l,precision):

    q1  = q_exp(q,-1)

    P = q1/q_exp(q,l*M)
    A = deepcopy(l*M)

    p = P.sum(axis = 1)
    s = P.sum(axis = 0)

    count = 0
    alpha = np.zeros(M.shape[0])
    beta = np.zeros(M.shape[1])

    while not (check(p,s,r,c,precision)) and count <= 3000:

        alpha = np.divide(p-r,np.sum(np.divide(P,(1+(1-q)*A)),axis = 1))
        A = (A.transpose() + alpha).transpose()

        P = q1/q_exp(q,A)
        s = P.sum(axis = 0)

        beta = np.divide(s-c,np.sum(np.divide(P,(1+(1-q)*A)),axis = 0))
        A += beta


        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)



        count +=1

    return P, count, q_obj(q,P,M,l)


#The TROT optimizing algorithm from the paper -- to use when q is in (0,1)
def second_order_sinkhorn(q,M,r,c,l,precision):

    n = M.shape[0]
    m = M.shape[1]
    q1  = q_exp(q,-1)

    A = deepcopy(l*M)

    #Add multipliers to make sure that A is not too large 
    #We subtract the minimum on columns and then on rows 
    #This does not change the solution of TROT 
    A = A-np.amin(A,axis = 0)
    A = (A.T-np.amin(A,axis = 1)).T
    
    P = q1/q_exp(q,A)

    p = P.sum(axis = 1)
    s = P.sum(axis = 0)

    count = 0
    alpha = np.zeros(M.shape[0])
    beta = np.zeros(M.shape[1])

    while not (check(p,s,r,c,precision)) and count <= 200000:


        A_q2 = np.divide(P,(1+(1-q)*A))
        a = (1-q/2)*(np.sum(np.divide(A_q2,(1+(1-q)*A)),axis = 1))
        b = - np.sum(A_q2,axis = 1)
        d = p-r
        delta = np.multiply(b,b) - 4*np.multiply(a,d)

        
        # Version 1 - Original
        #for i in range(n):
        #    if (delta[i] >=0 and d[i]<0 and a[i]>0):
        #        alpha[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
        #    elif (b[i] != 0):
        #        alpha[i] = 2*d[i]/(-b[i])
        #    else: alpha[i] = 0
        #    
        #    #Check that the multiplier is not too large
        #    if abs(alpha[i]) > 1/((2-2*q)*max(1+(1-q)*A[i,:])):
        #        alpha[i] = sign(d[i])*1/((2-2*q)*max(1+(1-q)*A[i,:]))

        # Negative p_{ij} safe
        for i in range(n):
            
            if   (delta[i] >= 0 and d[i]<0 and a[i]>0):
                alpha[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
            #elif (delta[i] <= 0 and d[i]>0 and a[i]>0):
            #    alpha[i] = - (b[i] - math.sqrt(abs(delta[i])))/(2*a[i])
            elif (b[i] != 0):
                alpha[i] = 2*d[i]/(-b[i])
            else: alpha[i] = 0
            
            #Check that the multiplier is not too large
            if abs(alpha[i]) > 1/((2-2*q)*max(1+(1-q)*A[i,:])):
                alpha[i] = sign(d[i])*1/((2-2*q)*max(1+(1-q)*A[i,:]))

        A = (A.transpose() + alpha).transpose()


        P = q1/q_exp(q,A)
        
        s = P.sum(axis = 0)

        A_q2 = np.divide(P,(1+(1-q)*A))
        a = (1-q/2)*(np.sum(np.divide(A_q2,(1+(1-q)*A)),axis = 0))
        b = - np.sum(A_q2,axis = 0)
        d = s - c
        delta = np.multiply(b,b) - 4*np.multiply(a,d)

        # Version 1 - Original
        #for i in range(m):
        #    if (delta[i] >=0 and d[i]<0 and a[i]>0):
        #        beta[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
        #    elif (b[i] != 0):
        #        beta[i] = 2*d[i]/(-b[i])
        #    else: beta[i] = 0
        #
        #    #Check that the multiplier is not too large
        #    if abs(beta[i]) > 1/((2-2*q)*max(1+(1-q)*A[:,i])):
        #        beta[i] = sign(d[i])*1/((2-2*q)*max(1+(1-q)*A[:,i]))

        # Version 2 - -ve pdf
        for i in range(m):
            if (delta[i] >=0 and d[i]<0 and a[i]>0):
                beta[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
            #elif (delta[i] <=0 and d[i]>0 and a[i]>0):
            #    beta[i] = - (b[i] - math.sqrt(abs(delta[i])))/(2*a[i])
            elif (b[i] != 0):
                beta[i] = 2*d[i]/(-b[i])
            else: beta[i] = 0
        
            #Check that the multiplier is not too large
            if abs(beta[i]) > 1/((2-2*q)*max(1+(1-q)*A[:,i])):
                beta[i] = sign(d[i])*1/((2-2*q)*max(1+(1-q)*A[:,i]))
        A += beta


        P = q1/q_exp(q,A)
        
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)

        count +=1
        if count % 10000 == 0:
            print(f'Iterations of second order sinkhorn: {count}')
    
    #print(P.sum())

    return P, count, q_obj(q,P,M,l)

#Less efficient than the previous rule...
def third_order_sinkhorn(q,M,r,c,l,precision):

    n = M.shape[0]
    m = M.shape[1]
    q1  = q_exp(q,-1)
    
    P = q1/q_exp(q,l*M)
    A = deepcopy(l*M)
    
    p = P.sum(axis = 1)
    s = P.sum(axis = 0)
    
    count = 0
    alpha = np.zeros(M.shape[0])
    beta = np.zeros(M.shape[1])

    while not (check(p,s,r,c,precision)) and count <= 3000:
        
        
        A_q1 = np.divide(P,(1+(1-q)*A))
        a = (2-q)*(np.sum(np.divide(A_q1,(1+(1-q)*A)),axis = 1))
        b = np.sum(A_q1,axis = 1)
        d = r-p
        delta = np.multiply(b,b) + 4*np.multiply(a,d) #This + sign is correct, since it should be -d and not d
        
        
        for i in range(n):
            if (delta[i] >=0 and d[i]>0):
                # when p < r this is the negative root, 
                # when p > r this is the smallest positive root when it exists
                #alpha[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
                alpha[i] = (-b[i] + math.sqrt(delta[i]))/(2*a[i])
            elif (b[i] != 0):
                alpha[i] = d[i]/b[i]
                #alpha[i] = 2*d[i]/(b[i]) #derived from auxiliary function expansion
            else: alpha[i] = 0
                
        A = (A.transpose() - alpha).transpose()
                
        
        P = q1/q_exp(q,A)
        s = P.sum(axis = 0)
        
        A_q1 = np.divide(P,(1+(1-q)*A))
        a = (2-q)*(np.sum(np.divide(A_q1,(1+(1-q)*A)),axis = 0))
        b = np.sum(A_q1,axis = 0)
        d = c-s
        delta = np.multiply(b,b) + 4*np.multiply(a,d)
        
        
        for i in range(m):
            if (delta[i] >=0 and d[i]>0):
                # when s < c this is the negative root, 
                # when s > c this is the smallest positive root when it exists
                beta[i] = (-b[i] + math.sqrt(delta[i]))/(2*a[i])
            elif (b[i] != 0):
                beta[i] = d[i]/b[i]
                #beta[i] = 2*d[i]/(b[i]) #derived from auxiliary function expansion
            else: beta[i] = 0
        A -= beta
    
        
        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)

        count +=1
    
    print(count)
    return P, count, q_obj(q,P,M,l)    
    
#Kullback-Leibler projected gradient method, to use when q > 1
def KL_proj_descent(q,M,r,c,l ,precision,T , rate = None, rate_type = None):
    
    if (q<=1): print("Warning: Projected gradient methods only work when q>1")    
    
    omega = math.sqrt(2*np.log(M.shape[0]))
    ind_map = np.asarray(np.matrix(r).transpose().dot(np.matrix(c)))   
    
    P = deepcopy(ind_map)

    new_score = q_obj(q,P,M,l)
    scores = []
    scores.append(new_score)
    
    best_score = new_score
    best_P= P
    
    count = 1    
    
    while count<=T: 
        
        G = q_grad(q,P,M,l)        
        
        
        if rate is None:
            tmp = np.exp(G*(-omega/(math.sqrt(T)*np.linalg.norm(G,np.inf)))) #Absolute horizon
            #tmp = np.exp(G*(-omega/(math.sqrt(count)*np.linalg.norm(G,np.inf)))) #Rolling horizon
        elif rate_type is "constant":
            tmp = np.exp(G*(-rate))
        elif rate_type is "constant_length":
            tmp = np.exp(G*(-rate*np.linalg.norm(G,np.inf)))
        elif rate_type is "diminishing":
            tmp = np.exp(G*(-rate/math.sqrt(count)))
        elif rate_type is "square_summable":
            tmp = np.exp(G*(-rate/count))
        
            
        P = np.multiply(P,tmp)
        P = Sinkhorn(P,r,c,precision)
        
        #Update score list
        new_score = q_obj(q,P,M,l)
        scores.append(new_score)  
        
        #Keep track of the best solution so far
        if (new_score < best_score):
            best_score = new_score
            best_P = P
        

        count+=1

    return best_P, scores
