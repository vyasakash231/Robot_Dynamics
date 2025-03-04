import numpy as np
import numpy.linalg as LA
from math import *

from .utils import *

class Robot_KM:
    def __init__(self,n,alpha,a,d,d_nn,joint_limits):
        self.n = n
        self.alpha = alpha
        self.a = a
        self.d = d
        self.d_nn = d_nn
        self.joint_limits = joint_limits

        self.step = 0

        self.q_range = np.zeros((self.n, 2))
        self.q_range[:,0] = self.joint_limits["lower"]
        self.q_range[:,1] = self.joint_limits["upper"]

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
        self.q = theta
        _, _, P_00 = self._transformation_matrix(theta)
        self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
        
        # Initial Referance State (Joint Space)
        self.qr = np.zeros(self.n)
        self.qr_dot = np.zeros(self.n)

    def J(self, theta):
        R, O, _ = self._transformation_matrix(theta)

        R_n_0 = R[self.n-1,:,:]
        O_n_0 = O[:,[self.n-1]]
        O_E_n = self.d_nn 
        O_E = O_n_0 + np.dot(R_n_0,O_E_n)

        Jz = np.zeros((3,self.n))
        Jw = np.zeros((3,self.n))

        for i in range(self.n):
            Rm = R[i,:,:]
            Z_i_0 = Rm[:,[2]]
            O_i_0 = O[:,[i]]
            O_E_i_0 = O_E - O_i_0
            
            cross_prod = np.cross(Z_i_0, O_E_i_0, axis=0)
            
            Jz[:,i] = cross_prod.reshape(-1)   # conver 2D of shape (3,1) to 1D of shape (3,)
            Jw[:,i] = Z_i_0.reshape(-1)   # conver 2D of shape (3,1) to 1D of shape (3,)

        jacobian = np.concatenate((Jz,Jw),axis=0)
        return jacobian.astype(np.float64), Jz.astype(np.float64), Jw.astype(np.float64)

    def J_dot(self, theta, theta_dot, H=None):
        """ https://doi.org/10.48550/arXiv.2207.01794 """
        if H is None:
            H = self.Hessian(theta)

        J_dot = np.zeros((6,self.n))
        
        for i in range(self.n):
            J_dot[:,[i]] = H[i,:,:].T @ theta_dot[:, np.newaxis]
        Jz_dot = J_dot[:3,:]
        Jw_dot = J_dot[3:,:]
        return J_dot.astype(np.float64), Jz_dot.astype(np.float64), Jw_dot.astype(np.float64)

    # only for Revolute joints
    def Hessian(self, theta):
        """ 
        Hessian_v = [H_1; H_2; ... ; H_6] = [(nxn)_1; (nxn)_2; ... ; (nxn)_6], where, H_i -> ith stacks of (n,n) matrix,
        Eqn (37) from this paper - https://doi.org/10.1109/CIRA.2005.1554272
        """    
        H = np.zeros((self.n, self.n, 6))  #  last index in Hessian_v is stack

        R, O, _ = self._transformation_matrix(theta)

        R_n_0 = R[self.n-1,:,:]
        O_n_0 = O[:,[self.n-1]]
        O_E_n = self.d_nn 
        O_E_0 = O_n_0 + np.dot(R_n_0,O_E_n)
        
        for i in range(self.n):
            Ri = R[i,:,:]
            Z_i_0 = Ri[:,[2]]
            for j in range(self.n):
                Rj = R[j,:,:]
                Z_j_0 = Rj[:,[2]]
                O_j_0 = O[:,[j]]
                O_E_j_0 = O_E_0 - O_j_0

                if i <= j:
                    cross_prod_j = np.cross(Z_j_0, O_E_j_0, axis=0)
                    H_z = np.cross(Z_i_0, cross_prod_j, axis=0)

                    if i != j:
                        H_w = np.cross(Z_i_0, Z_j_0, axis=0)
                    else:
                        H_w = np.zeros((3,1))

                    H[i,j,:] = np.concatenate((H_z.reshape(-1), H_w.reshape(-1)))
                else:
                    H[i,j,:] = H[j,i,:].copy()
        return H
    
    def manipulability_Jacobian(self, theta, J, H=None):
        """ https://doi.org/10.48550/arXiv.2207.01794 """
        if H is None:
            H = self.Hessian(theta)

        # Calculate the manipulability of the robot
        manipulability = np.sqrt(LA.det(J @ J.T))

        row, column = J.shape

        J_m = np.zeros((self.n,1))
        for i in range(self.n):
            column_1 = J @ H[i,:,:row]
            column_2 = LA.inv(J @ J.T)
            
            # Reshape into a 9x1 vector (column-wise stacking)
            column_1 = column_1.flatten("F") # 'F' means column-major order (Fortran-style)
            column_2 = column_2.flatten("F") # 'F' means column-major order (Fortran-style)
            
            J_m[i] = manipulability * (column_1.T @ column_2)
        return J_m

    def taskspace_coord(self,theta):
        _, O, P_00 = self._transformation_matrix(theta)

        X_cord = np.concatenate(([0],O[0,:],P_00[[0],0]))
        Y_cord = np.concatenate(([0],O[1,:],P_00[[1],0]))
        Z_cord = np.concatenate(([0],O[2,:],P_00[[2],0]))
        return X_cord, Y_cord, Z_cord        

    def FK(self, theta, theta_dot, theta_ddot, level="vel"):
        if level == "vel":
            _,J,_ = self.J(theta)   # only linear velocity

            if theta_dot.ndim != 2:
                theta_dot = theta_dot.reshape((self.n, 1))

            Xe_dot = J @ theta_dot   # end-effector velocity
            _, _, P_00 = self._transformation_matrix(theta)  # end-effector position
            self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
            return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), []
        
        if level == "acc":
            _,J,_ = self.J(theta)   # only linear velocity

            _,J_dot,_ = self.J_dot(theta, theta_dot)

            if theta_dot.ndim != 2:
                theta_dot = theta_dot.reshape((self.n, 1))
            if theta_ddot.ndim != 2:
                theta_ddot = theta_ddot.reshape((self.n, 1))

            Xe_dot = J @ theta_dot    # end-effector velocity
            Xe_ddot = J @ theta_ddot + J_dot @ theta_dot    # end-effector acceleration
            _, _, P_00 = self._transformation_matrix(theta)  # end-effector position
            self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
            return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), Xe_ddot.astype(np.float64)      

    # TODO: complete the Inverse Kinematics code
    def IK(self, Xd, X_dot, method=1):
        Ex = np.array(Xd - self.Xe, dtype=float)

        if method == 1:
            if self.step == 0:
                self.e0 = Ex
            
            Jc = np.eye((self.n))   # Jacobian for additional task

            We, Wc, Wv = weights_1(3, self.n, self.q_range, self.q)

            _,Je,_ = self.J(self.q)   # only linear velocity

            Jn = LA.inv(Je.T @ We @ Je + Jc.T @ Wc @ Jc + Wv) @ Je.T @ We
            
            # Adaptive gain
            K = 5 * (1-np.exp(-self.step*0.01)) * np.exp((LA.norm(self.e0) - LA.norm(Ex)) / LA.norm(Ex))
            
            q_dot = K * (Jn @ X_dot)

            self.q += q_dot.reshape(-1)

            Ex = Xd - self.Xe

            self.step += 1
            return self.q
        
        if method == 2:
            while LA.norm(Ex) > 0.002:  # 2.0mm 
                i += 1
        
        if method == 3:
            Wn = np.ones((self.n, self.n))  # diagonal damping matrix
            while LA.norm(Ex) > 0.002:  # 2.0mm 
                _,J,_ = self.J(self.q)   # only linear velocity

                J_pinv = LA.pinv(J)
                Jm = self.manipulability_Jacobian(self.q)

                q_null = (np.eye(self.n) - J_pinv @ J) @ Jm

                A = J.T @ We @ J + Wn
                g = J.T @ We @ Ex[:, np.newaxis] + q_null
                q_dot = J_pinv @ Ex[:, np.newaxis] + LA.inv(A) @ g

                self.q += q_dot.reshape(-1)

                Ex = Xd - self.Xe
            return self.q
    
