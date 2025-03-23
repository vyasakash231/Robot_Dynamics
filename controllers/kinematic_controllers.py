import os
import sys
import numpy as np
import numpy.linalg as LA
from math import *

class Kinematic_Control:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.qr_dot_t1 = np.zeros(self.robot.n)

    def pseudo_inverse(self, J):
        J_inv = J.T @ LA.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian
        return J_inv

    def damped_least_square_with_JLA_1(self, Xd, X_dot):
        q = self.robot.robot_KM.IK(Xd, X_dot, method=1)
        return q.reshape(-1)
    
    def pd_osc(self, Ex, Ex_dot, Kp, Kd):
        # PD Control Law
        Xd_ddot = Kp @ Ex + Kd @ Ex_dot
        return Xd_ddot

    def pd_dmp(self, Xdmp, Xdmp_dot, Xe, Xe_dot, Kp, Kd):
        # PD Control Law
        Ex = Xdmp - Xe
        Ex_dot = Xdmp_dot - Xe_dot

        Xd_ddot = Kp @ Ex + Kd @ Ex_dot 
        return Xd_ddot
    
    def pd_gp(self, Xgp, Xgp_dot, Xe, Xe_dot, Kp, Kd):
        # PD Control Law
        Ex = Xgp - Xe
        Ex_dot = Xgp_dot - Xe_dot

        Xd_ddot = Kp @ Ex + Kd @ Ex_dot 
        return Xd_ddot

    """Simplified Acceleration-based Control Variation 1 (With Null Space Pre-multiplication of M)"""
    def acceleration_based_control_1(self, q, q_dot, Ex, Ex_dot, Xd_ddot, Kp_ts, Kd_ts, kd_js):
        # jacobian matrix
        _,J,_ = self.robot.robot_KM.J(q)    # J:(6,n), Jv:(3,n), Jw:(3,n)

        # Hessian Matrix
        H = self.robot.robot_KM.Hessian(q)
        
        # Manipulability Jacobian matrix
        # J_m = self.robot.robot_KM.manipulability_Jacobian(q, J, H)

        """
        A simple optimization criterion for redundancy resolution, with a cost function:
                    g(q) = (1/2) * (q - q0)^T * Kw * (q - q0)
                    del_g = Kw * (q - q0)  => J_m
        """
        Kw = np.eye(self.robot.n)
        J_m = Kw @ (q - self.robot.q)[:, np.newaxis]

        # jacobian_dot matrix
        _,J_dot,_ = self.robot.robot_KM.J_dot(q, q_dot, H)    # J_dot, Jv_dot, Jw_dot

        J_pinv = np.linalg.pinv(J)    # pseudoinverse of the Jacobian

        """Eqn (26) from https://doi.org/10.1177/0278364908091463"""
        # Xd_ddot is the desired task space acceleration, Xd_dot is the desired task space velocity, Xd is the desired task space position
        Xr_ddot = Xd_ddot + Kd_ts @ Ex_dot + Kp_ts @ Ex 

        """Eqn (40) from https://doi.org/10.1177/0278364908091463"""
        zeta = -(kd_js @ q_dot[:, np.newaxis] + 50 * J_m)

        """Eqn (39) from https://doi.org/10.1177/0278364908091463"""
        qr_ddot = J_pinv @ (Xr_ddot - J_dot @ q_dot[:, np.newaxis]) + (np.eye(self.robot.n) - J_pinv @ J) @ zeta   # In radians/s^2
        qr_ddot = qr_ddot.reshape(-1)

        # Numerical Differentation
        # self.robot.qr_dot = self.robot.qr_dot + qr_ddot * self.robot.dt  # In radians/s

        # Numerical Differentation
        self.robot.qr = self.robot.qr + self.robot.qr_dot * self.robot.dt  # In radians
        return self.robot.qr.astype(np.float64), qr_ddot.astype(np.float64)
        
    """Simplified Acceleration-based Control Variation 2 (Without Null Space Pre-multiplication of M)"""
    def acceleration_based_control_2(self, q, q_dot, Ex, Ex_dot, Xd_ddot, Kp_ts, Kd_ts):
        # jacobian matrix
        _,J,_ = self.robot.robot_KM.J(q)    # J:(6,n), Jv:(3,n), Jw:(3,n)

        # jacobian_dot matrix
        _,J_dot,_ = self.robot.robot_KM.J_dot(q, q_dot)     # J_dot, Jv_dot, Jw_dot

        J_pinv = np.linalg.pinv(J)     # pseudoinverse of the Jacobian

        """Eqn (26) from https://doi.org/10.1177/0278364908091463"""
        # Xd_ddot is the desired task space acceleration, Xd_dot is the desired task space velocity, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        # Ex_dot = Xd_dot - Xe_dot is the task-space velocity error, where, Xe is the end-effector velocity
        Xr_ddot = Xd_ddot + Kd_ts @ Ex_dot + Kp_ts @ Ex 

        """Eqn (45) from https://doi.org/10.1177/0278364908091463"""
        qr_ddot = J_pinv @ (Xr_ddot - J_dot @ q_dot[:, np.newaxis])    # In radians/s^2
        qr_ddot = qr_ddot.reshape(-1)

        # Numerical Differentation
        self.robot.qr_dot = self.robot.qr_dot + qr_ddot * self.robot.dt  # In radians/s

        # Numerical Differentation
        self.robot.qr = self.robot.qr + self.robot.qr_dot * self.robot.dt  # In radians
        return self.robot.qr.astype(np.float64), self.robot.qr_dot.astype(np.float64), qr_ddot.astype(np.float64)
        