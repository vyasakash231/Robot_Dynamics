from math import *
import numpy as np


def apply_joint_limits(q, q_dot, q_limits, k_spring=2.5, b_damper=0.5):
    """Add spring-damper forces when near limits"""
    q_min, q_max = q_limits["lower"], q_limits["upper"]
    limit_force = np.zeros_like(q)
    
    for i in range(len(q)):
        if q[i] < q_min[i]:
            # Spring force + damping force when below minimum
            limit_force[i] = k_spring*(q_min[i] - q[i]) - b_damper*q_dot[i]
        elif q[i] > q_max[i]:
            # Spring force + damping force when above maximum
            limit_force[i] = k_spring*(q_max[i] - q[i]) - b_damper*q_dot[i]
    return limit_force

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def weights_1(m,n,q_range,q):
    epsilon = 0.5 * np.ones(n)  # Activation buffer region width
    
    const = 300
    We = np.zeros((m,m))
    for i in range(0,m):
        We[i,i] = 50

    Wc = np.zeros((n,n))
    for i in range(0,n):
        if q[i] < q_range[i,0]:
            Wc[i,i] = const
        elif q_range[i,0] <= q[i] <= (q_range[i,0] + epsilon[i]):
            Wc[i,i] = (const/2)*(1 + cos(pi*((q[i] - q_range[i,0])/epsilon[i])))
        elif (q_range[i,0] + epsilon[i]) < q[i] < (q_range[i,1] - epsilon[i]):
            Wc[i,i] = 0
        elif (q_range[i,1] - epsilon[i]) <= q[i] <= q_range[i,1]:
            Wc[i,i] = (const/2)*(1 + cos(pi*((q_range[i,1] - q[i])/epsilon[i])))
        else:
            Wc[i,i] = const

    Wv = np.zeros((n,n))
    for i in range(0,n):
        Wv[i,i] = 0.5
    return We, Wc, Wv

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def cost_func(n,K,q,q_range,m):
    # Initiate
    c = np.zeros((n,))
    b = np.zeros((n,))
    del_phi_del_q = np.zeros((n,))
    q_c = np.mean(q_range,axis = 1); # column vector containing the mean of each row
    del_q = q_range[:,1] - q_range[:,0]; # Total working range of each joint

    for i in range(0,n):
        if q[0,i] >= q_c[i]:
            c[i] = pow((K[i,i]*((q[0,i] - q_c[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q[0,i] - q_c[i])/del_q[i])),m-1)
        elif q[0,i] < q_c[i]:
            c[i] = pow((K[i,i]*((q_c[i] - q[0,i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q_c[i] - q[0,i])/del_q[i])),(m-1))

    L = np.sum(c)

    for j in range(0,n):
        if q[0,j] >= q_c[j]:
            del_phi_del_q[j] = pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])
        elif q[0,j] < q_c[j]:
            del_phi_del_q[j] = -pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])

    v = -del_phi_del_q
    return v

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
