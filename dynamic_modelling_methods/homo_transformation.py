from math import *
import numpy as np
from sympy import *

def symbolic_transformation_matrix(n,alpha,a,d,theta,d_nn):
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

    T_final = I
    d_nn = np.append(d_nn, 1).reshape((4,1)) # make the vector homogeneous and column vector
    P_00_homo = T_final * Matrix(d_nn)
    O_E_0 = P_00_homo[0:3,:]  # convert back to Eucledian form   
    
    a_n_0[:,:n] = O_n_0[:,1:] - O_n_0[:,:-1]  # coord of (i)th joint wrt (i-1)th joint, a_i_0 = a_{i-1,i}_0
    a_n_0[:,-1] = O_E_0 - O_n_0[:,[-1]]
    
    return (R, O_n_0, a_n_0)

#############################################################################################################################

def forward_kinematics(n,alpha,a,d,theta,d_nn):
    I = np.eye(4)
    R = np.zeros((n,3,3))
    O = np.zeros((3,n+1))

    # Transformation Matrix
    for i in range(0,n):
        T = np.array([[      cos(theta[i])        ,      -sin(theta[i])        ,        0      ,        a[i]        ],
                      [sin(theta[i])*cos(alpha[i]), cos(theta[i])*cos(alpha[i]), -sin(alpha[i]), -d[i]*sin(alpha[i])],                                               
                      [sin(theta[i])*sin(alpha[i]), cos(theta[i])*sin(alpha[i]),  cos(alpha[i]),  d[i]*cos(alpha[i])],     
                      [             0             ,             0              ,        0      ,          1         ]])

        T_new = np.dot(I,T)
        R[i,:,:] = T_new[:3,:3]
        O[:3,i+1] = T_new[:3,-1]
        I = T_new

    T_final = I
    d_nn = np.append(d_nn, 1).reshape((4,1)) # make the vector homogeneous and column vector
    P_00_homo = T_final @ d_nn
    P_00 = P_00_homo[0:3,:]  # convert back to Eucledian form

    X_cord = np.concatenate((O[0,:], P_00[[0],0]))
    Y_cord = np.concatenate((O[1,:], P_00[[1],0]))
    Z_cord = np.concatenate((O[2,:], P_00[[2],0]))
    return(X_cord,Y_cord,Z_cord)

#############################################################################################################################

def symbolic_Jacobian(n,alpha,a,d,theta,d_nn):
    R, O, _ = symbolic_transformation_matrix(n, alpha, a, d, theta, d_nn)

    R_n_0 = R[n-1,:,:]
    O_n_0 = np.transpose(np.array([O[:,n-1]]))
    O_E_n = d_nn 
    O_E = O_n_0 + np.dot(R_n_0,O_E_n)

    Jz = np.zeros((3,n))
    Jw = np.zeros((3,n))

    for i in range(n):
        Z_i_0 = np.transpose(np.array([R[i,:,2]]))
        O_i_0 = np.transpose(np.array([O[:,i]]))
        O_E_i_0 = O_E - O_i_0

        cross_prod = np.cross(Z_i_0, O_E_i_0, axis=0)

        # Linear
        Jz[:,i] = np.reshape(cross_prod,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)  
        
        # Angular
        Jw[:,i] = np.reshape(Z_i_0,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)

    J = np.concatenate((Jz,Jw),axis=0)
    return Jz, Jw