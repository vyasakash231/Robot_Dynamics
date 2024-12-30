from math import *
import numpy as np
from sympy import *


def symbolic_transformation_matrix(n,alpha,a,d,theta,d_nn=None):
    I = eye(4)
    R = [0 for _ in range(n)]  # list of rotation matrices
    O_n_0 = zeros(3,n+1)  # adding base coordinates  (O_0, O_1, O_2, ...., O_n)
    a_n_0 = zeros(3,n+1)

    # Transformation Matrix
    for i in range(0,n):
        T = Matrix([[      cos(theta[i])        ,      -sin(theta[i])        ,        0      ,        a[i]        ],
                    [sin(theta[i])*cos(alpha[i]), cos(theta[i])*cos(alpha[i]), -sin(alpha[i]), -d[i]*sin(alpha[i])],                                               
                    [sin(theta[i])*sin(alpha[i]), cos(theta[i])*sin(alpha[i]),  cos(alpha[i]),  d[i]*cos(alpha[i])],     
                    [             0             ,             0              ,        0      ,          1         ]])

        T_new = I * T
        R[i] = T_new[:3,:3]
        O_n_0[:3,i+1] = T_new[:3,-1]  # coord of (i)th joint wrt (0)th joint
        I = T_new

    if d_nn is not None:
        T_final = I
        d_nn = np.append(d_nn, 1).reshape((4,1)) # make the vector homogeneous and column vector
        P_00_homo = T_final * Matrix(d_nn)
        O_E_0 = P_00_homo[0:3,:]  # convert back to Eucledian form   
        
        a_n_0[:,:n] = O_n_0[:,1:] - O_n_0[:,:-1]  # coord of (i)th joint wrt (i-1)th joint, a_i_0 = a_{i-1,i}_0
        a_n_0[:,-1] = O_E_0 - O_n_0[:,[-1]]    
        return R, O_n_0, a_n_0
    else:
        return R, O_n_0

#############################################################################################################################
"""for revolute and prismatic joints"""
def symbolic_Jacobian(n, alpha, a, d, theta, epsilon, d_nn=None):
    R, O = symbolic_transformation_matrix(n, alpha, a, d, theta, d_nn=None)

    R_n_0 = R[-1]
    O_n_0 = O[:,[n]]
    O_E_n = d_nn 

    O_E_0 = O_n_0 + R_n_0*O_E_n

    Jz = zeros(3,n)
    Jw = zeros(3,n)

    for i in range(n):
        Rm = R[i] 
        Z_i_0 = Rm[:,[-1]]
        O_i_0 = O[:,[i+1]]
        O_E_i_0 = O_E_0 - O_i_0

        cross_prod = Z_i_0.cross(O_E_i_0)

        Jz[:,i] = epsilon[i]*cross_prod + (1-epsilon[i])*Z_i_0   # Linear
        Jw[:,i] = epsilon[i]*Z_i_0   # Angular

    J = Jz.col_join(Jw)
    return simplify(J)  # if visual is needed then use print(pretty(J))

#############################################################################################################################
"""for revolute joints only"""
def symbolic_Hessian(n,alpha,a,d,epsilon,d_nn=None):
    """ Hessian_v = [H_1; H_2; ... ; H_6] = [(nxn)_1; (nxn)_2; ... ; (nxn)_6], where, H_i -> ith stacks of (n,n) matrix """    
    H = []  #  last index in Hessian_v is stack
    H_i = zeros(n,n)  # (n,n) matrix

    theta = [symbols(f'q{i+1}') for i in range(n)]
    
    J = symbolic_Jacobian(n,alpha,a,d,theta,epsilon,d_nn)  # (6,n) matrix
    
    for i in range(6):
        for j in range(n):
            for k in range(n):
                H_i[j,k] = diff(J[i,k], theta[j])
        H.append(H_i.copy())
    return H


if __name__ == "__main__":
    # DH-Parameters
    # n = 4  # DOF (No of Joint)
    # alpha = np.radians([0,90,0,0])  # In radians
    # a = np.array([0,0,np.sqrt(0.128**2 + 0.024**2),0.124])  # in meters
    # d = np.array([0.077,0,0,0])  # in meters
    # d_nn = Matrix([[0.126], [0], [0]])  # coord of EE wrt to last joint frame in meters
    # epsilon = np.array([1,1,1,1])  # 1 for revolute and 0 for prismatic
    
    l1, l2 = Symbol('l1'), Symbol('l2')

    n = 2  # DOF (No of Joint)
    alpha = np.radians([0,0])  # In radians
    a = Matrix([0,l1])  # in meters
    d = Matrix([0,0])  # in meters
    d_nn = Matrix([[l2], [0], [0]])  # coord of EE wrt to last joint frame in meters
    epsilon = np.array([1,1])  # 1 for revolute and 0 for prismatic

    H = symbolic_Hessian(n,alpha,a,d,epsilon,d_nn)
    # print(pretty(H))    


