import numpy as np
from math import *

class Robot_KM:
    def __init__(self,n,alpha,a,d,d_nn):
        self.n = n
        self.alpha = alpha
        self.a = a
        self.d = d
        self.d_nn = d_nn

    def _transformation_matrix(self,theta):
        I = np.eye(4)
        R = np.zeros((self.n,3,3))
        O = np.zeros((3,self.n))

        # Transformation Matrix
        for i in range(self.n):
            T = np.array([[         cos(theta[i])          ,          -sin(theta[i])         ,           0        ,          self.a[i]           ],
                          [sin(theta[i])*cos(self.alpha[i]), cos(theta[i])*cos(self.alpha[i]), -sin(self.alpha[i]), -self.d[i]*sin(self.alpha[i])],                                               
                          [sin(theta[i])*sin(self.alpha[i]), cos(theta[i])*sin(self.alpha[i]),  cos(self.alpha[i]),  self.d[i]*cos(self.alpha[i])],     
                          [               0                ,                 0               ,           0        ,               1              ]])

            T_new = np.dot(I,T)
            R[i,:,:] = T_new[0:3,0:3]
            O[0:3,i] = T_new[0:3,3]
            I = T_new
            i= i + 1

        T_final = I
        d_nn_extended = np.append(self.d_nn, [[1]], axis=0)
        P_00_home = np.dot(T_final,d_nn_extended)
        P_00 = P_00_home[0:3]
        return  R, O, P_00
    
    def initial_state(self, theta):
        _, _, P_00 = self._transformation_matrix(theta)
        self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
        
        # Initial Referance State (Joint Space)
        self.qr = np.zeros(self.n)
        self.qr_dot_t0 = np.zeros(self.n)

    def J(self, theta):
        R, O, _ = self._transformation_matrix(theta)

        R_n_0 = R[self.n-1,:,:]
        O_n_0 = np.transpose(np.array([O[:,self.n-1]]))
        O_E_n = self.d_nn 
        O_E = O_n_0 + np.dot(R_n_0,O_E_n)

        Jz = np.zeros((3,self.n))
        Jw = np.zeros((3,self.n))

        for i in range(self.n):
            Z_i_0 = np.transpose(np.array([R[i,:,2]]))
            O_i_0 = np.transpose(np.array([O[:,i]]))
            O_E_i_0 = O_E - O_i_0

            cross_prod = np.cross(Z_i_0, O_E_i_0, axis=0)
            Jz[:,i] = np.reshape(cross_prod,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)
            Jw[:,i] = np.reshape(Z_i_0,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)

        J = np.concatenate((Jz,Jw),axis=0)
        return Jz.astype(np.float64)
    
    def _f(self, i, Z, O):
        f = np.zeros((3,1))
        for j in range(i,self.n):
            f = f + np.cross(Z, O[:,[j+1]] - O[:,[j]], axis=0)
        return f
    
    def _g(self, i, Z, O):
        g = np.zeros((3,1))
        for j in range(i+1, self.n):
            g = g + np.cross(Z, O[:,[j+1]] - O[:,[j]], axis=0)
        return g

    # def J_dot(self, theta, theta_dot):
    #     R, O, _ = self._transformation_matrix(theta)

    #     R_n_0 = R[self.n-1,:,:]
    #     O_n_0 = np.transpose(np.array([O[:,self.n-1]]))  # O_n
    #     O_E_n = self.d_nn 
    #     O_E_0 = O_n_0 + np.dot(R_n_0,O_E_n)  # O_E = O_n+1

    #     # Add end-effector coord into O matrix
    #     O = np.hstack((O, O_E_0))  # [O_1, O_2, O_3, ... , O_n, O_E]

    #     Z_1_0 = np.transpose(np.array([R[0,:,2]]))  # first Z column -> Z_1
    #     Z_n_0 = np.transpose(np.array([R[-1,:,2]]))  # last Z column -> Z_n 

    #     Jz_dot = np.zeros((3,self.n))
    #     # Jw_dot = np.zeros((3,self.n))

    #     for i in range(self.n-1):
    #         Z_i_0 = np.transpose(np.array([R[i,:,2]]))  # Z_{i} in base frame
    #         Z_i_1_0 = np.transpose(np.array([R[i+1,:,2]]))  # Z_{i+1} in base frame
            
    #         Jz_dot[:,[i]] = theta_dot[i] * np.cross(Z_i_0, self._f(i, Z_i_0, O), axis=0) + \
    #                         theta_dot[i+1] * (np.cross(Z_i_0, self._f(i+1, Z_i_1_0, O), axis=0) + np.cross(Z_i_1_0, self._g(i, Z_i_0, O), axis=0))
        
    #     Jz_dot[:,[-1]] = theta_dot[-1] * np.cross(Z_n_0, self._f(i+1, Z_n_0, O), axis=0) + \
    #                     theta_dot[0] * np.cross(Z_n_0, np.cross(Z_1_0, (O_E_0 - O_n_0), axis=0), axis=0)
    #     return Jz_dot.astype(np.float64)

    def J_dot(self, theta, theta_dot, H=None):
        """ https://doi.org/10.48550/arXiv.2207.01794 """
        if H is None:
            H = self.Hessian_matrix(theta)

        Jz_dot = np.zeros((3,self.n))
        
        for i in range(self.n):
            Jz_dot[:,[i]] = H[:,:,i] @ theta_dot.reshape((self.n,1))
        return Jz_dot.astype(np.float64)

    def Hessian_matrix(self, theta):
        """ https://doi.org/10.48550/arXiv.2207.01794 """
        R, O, _ = self._transformation_matrix(theta)

        R_n_0 = R[self.n-1,:,:]
        O_n_0 = np.transpose(np.array([O[:,self.n-1]]))  # O_n
        O_E_n = self.d_nn 
        O_E_0 = O_n_0 + np.dot(R_n_0,O_E_n)  # O_E = O_n+1

        # Add end-effector coord into O matrix
        O = np.hstack((O, O_E_0))  # [O_1, O_2, O_3, ... , O_n, O_E]

        Z_1_0 = np.transpose(np.array([R[0,:,2]]))  # first Z column -> Z_1
        Z_n_0 = np.transpose(np.array([R[-1,:,2]]))  # last Z column -> Z_n 

        """ Hessian_v = [H_1; H_2; ... ; H_n] = [(3xn)_1; (3xn)_2; ... ; (3xn)_n], where, H_i -> ith stacks of (3,n) matrix """
        Hessian_v = np.zeros((3,self.n,self.n))  #  last index in Hessian_v is stack
        
        for i in range(self.n-1):
            Z_i_0 = np.transpose(np.array([R[i,:,2]]))  # Z_{i} in base frame
            Z_i_1_0 = np.transpose(np.array([R[i+1,:,2]]))  # Z_{i+1} in base frame
            
            Hessian_v[:,[i],i] = np.cross(Z_i_0, self._f(i, Z_i_0, O), axis=0) 
            Hessian_v[:,[i+1],i] = np.cross(Z_i_0, self._f(i+1, Z_i_1_0, O), axis=0) + np.cross(Z_i_1_0, self._g(i, Z_i_0, O), axis=0)
      
        Hessian_v[:,[-1],-1] = np.cross(Z_n_0, self._f(i+1, Z_n_0, O), axis=0) 
        Hessian_v[:,[0],-1] = np.cross(Z_n_0, np.cross(Z_1_0, (O_E_0 - O_n_0), axis=0), axis=0)
        return Hessian_v.astype(np.float64)
    
    def manipulability_Jacobian(self, theta, J=None, H=None):
        """ https://doi.org/10.48550/arXiv.2207.01794 """
        if J is None:
            J = self.J(theta)
        if H is None:
            H = self.Hessian_matrix(theta)

        # Calculate the manipulability of the robot
        manipulability = np.sqrt(np.linalg.det(J @ J.T))

        J_m = np.zeros((self.n,1))
        for i in range(self.n):
            column_1 = J @ H[:,:,i].T
            column_2 = np.linalg.inv(J @ J.T)
            
            # Reshape into a 9x1 vector (column-wise stacking)
            column_1 = column_1.flatten("F") # 'F' means column-major order (Fortran-style)
            column_2 = column_2.flatten("F") # 'F' means column-major order (Fortran-style)
            
            J_m[i] = manipulability * (column_1.T @ column_2)
        return J_m

    def FK(self,theta):
        _, O, P_00 = self._transformation_matrix(theta)

        X_cord = np.concatenate(([0],O[0,:],P_00[[0],0]))
        Y_cord = np.concatenate(([0],O[1,:],P_00[[1],0]))
        Z_cord = np.concatenate(([0],O[2,:],P_00[[2],0]))
        return X_cord, Y_cord, Z_cord        

    def kinematic_model(self, dt, theta, theta_dot, theta_ddot, level="vel"):
        if level == "vel":
            J = self.J(theta)

            if theta_dot.ndim != 2:
                theta_dot = theta_dot.reshape((self.n, 1))

            Xe_dot = J @ theta_dot   # end-effector velocity
            #self.Xe = self.Xe + Xe_dot * dt  # end-effector position
            _, _, P_00 = self._transformation_matrix(theta)
            self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
            return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), []
        
        if level == "acc":
            J = self.J(theta)
            J_dot = self.J_dot(theta, theta_dot)

            if theta_dot.ndim != 2:
                theta_dot = theta_dot.reshape((self.n, 1))
            if theta_ddot.ndim != 2:
                theta_ddot = theta_ddot.reshape((self.n, 1))

            Xe_dot = J @ theta_dot    # end-effector velocity
            Xe_ddot = J @ theta_ddot + J_dot @ theta_dot    # end-effector acceleration
            # self.Xe = self.Xe + Xe_dot * dt  # end-effector position
            _, _, P_00 = self._transformation_matrix(theta)
            self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
            return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), Xe_ddot.astype(np.float64)        
    
