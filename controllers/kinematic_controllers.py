import os
import sys
import numpy as np
from math import *


class Kinematic_Control:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.qr_dot_t1 = np.zeros(self.robot.n)

    def damped_least_square_with_JLA_1(self, Xd, X_dot):
        q = self.robot.robot_KM.IK(Xd, X_dot, method=1)
        return q.reshape(-1)
    
    def velocity_based_control_1(self, q, Ex, Xd_dot, Kp_ts):
        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J)

        # pseudoinverse of the Jacobian
        J_inv = J.T @ np.linalg.inv(J @ J.T)    

        """Eqn (10) from https://doi.org/10.1177/0278364908091463"""
        # Xd_dot is the desired task space velocities, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        Xr_dot = Xd_dot + Kp_ts @ Ex  # reference task space velocity
        
        Lambda = -10

        """Eqn (44) from https://doi.org/10.48550/arXiv.2207.01794 and Eqn (11) from https://doi.org/10.1177/0278364908091463"""
        qr_dot_t1 = J_inv @ Xr_dot + (1/Lambda) * (np.eye(self.robot.n) - J_inv @ J) @ J_m   # In radians/s  (reference joint space velocity)
        qr_dot_t1 = qr_dot_t1.reshape(-1)

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + qr_dot_t1 * self.robot.dt  # In radians

        # Numerical Integration
        qr_ddot = (qr_dot_t1 - self.robot.robot_KM.qr_dot) / self.robot.dt     # this is creating much smoother torque
        self.robot.robot_KM.qr_dot = qr_dot_t1

        return self.robot.robot_KM.qr.astype(np.float64), qr_dot_t1.astype(np.float64), qr_ddot.astype(np.float64)
    
    def velocity_based_control_2(self, q, Ex, Xd_dot, Kp_ts):
        self.robot.robot_KM.qr_dot = self.qr_dot_t1.reshape(-1)

        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J)

        # pseudoinverse of the Jacobian
        J_pinv = J.T @ np.linalg.inv(J @ J.T)    

        """Eqn (10) from https://doi.org/10.1177/0278364908091463"""
        # Xd_dot is the desired task space velocities, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        Xr_dot = Xd_dot + Kp_ts @ Ex  # reference task space velocity
        
        Lambda = -5

        """Eqn (44) from https://doi.org/10.48550/arXiv.2207.01794 and Eqn (11) from https://doi.org/10.1177/0278364908091463"""
        self.qr_dot_t1 = J_pinv @ Xr_dot + (1/Lambda) * (np.eye(self.robot.n) - J_pinv @ J) @ J_m   # In radians/s  (reference joint space velocity)

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + self.robot.robot_KM.qr_dot * self.robot.dt  # In radians

        qr_ddot = np.zeros((self.robot.n))     # this is creating much smoother torque
        return self.robot.robot_KM.qr.astype(np.float64), self.robot.robot_KM.qr_dot.astype(np.float64), qr_ddot.astype(np.float64)
         
    def acceleration_based_control_1(self, q, q_dot, Ex, Ex_dot, Xd_ddot, Kp_task_space, Kd_task_space):
        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Hessian Matrix
        H = self.robot.robot_KM.Hessian(q)

        # jacobian_dot matrix
        J_dot = self.robot.robot_KM.J_dot(q, q_dot, H)

        # pseudoinverse of the Jacobian
        J_inv = J.T @ np.linalg.inv(J @ J.T)    

        """Eqn (26) from https://doi.org/10.1177/0278364908091463"""
        # Xd_ddot is the desired task space acceleration, Xd_dot is the desired task space velocity, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        # Ex_dot = Xd_dot - Xe_dot is the task-space velocity error, where, Xe is the end-effector velocity
        Xr_ddot = Xd_ddot + Kd_task_space @ Ex_dot + Kp_task_space @ Ex 

        """Eqn (45) from https://doi.org/10.1177/0278364908091463"""
        qr_ddot = J_inv @ (Xr_ddot - J_dot @ q_dot[:, np.newaxis])  # In radians/s^2
        qr_ddot = qr_ddot.reshape(-1)

        # Numerical Differentation
        self.robot.robot_KM.qr_dot = self.robot.robot_KM.qr_dot + qr_ddot * self.robot.dt  # In radians/s

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + self.robot.robot_KM.qr_dot * self.robot.dt  # In radians
        return self.robot.robot_KM.qr.astype(np.float64), qr_ddot.astype(np.float64)
        
    """Simplified Acceleration-based Control Variation 1 (With Null Space Pre-multiplication of M)"""
    def acceleration_based_control_2(self, q, q_dot, Ex, Ex_dot, Xd_ddot, Kp_task_space, Kd_task_space, kd_joint_space):
        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Hessian Matrix
        H = self.robot.robot_KM.Hessian(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J, H)

        # jacobian_dot matrix
        J_dot = self.robot.robot_KM.J_dot(q, q_dot, H)

        J_inv = J.T @ np.linalg.inv(J @ J.T)    # pseudoinverse of the Jacobian

        """Eqn (26) from https://doi.org/10.1177/0278364908091463"""
        # Xd_ddot is the desired task space acceleration, Xd_dot is the desired task space velocity, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        # Ex_dot = Xd_dot - Xe_dot is the task-space velocity error, where, Xe is the end-effector velocity
        Xr_ddot = Xd_ddot + Kd_task_space @ Ex_dot + Kp_task_space @ Ex 

        """Eqn (40) from https://doi.org/10.1177/0278364908091463"""
        zeta = -(kd_joint_space @ q_dot[:, np.newaxis] + 30 * J_m)

        """Eqn (39) from https://doi.org/10.1177/0278364908091463"""
        qr_ddot = J_inv @ (Xr_ddot - J_dot @ q_dot[:, np.newaxis]) + (np.eye(self.robot.n) - J_inv @ J) @ zeta   # In radians/s^2
        qr_ddot = qr_ddot.reshape(-1)

        # Numerical Differentation
        self.robot.robot_KM.qr_dot = self.robot.robot_KM.qr_dot + qr_ddot * self.robot.dt  # In radians/s

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + self.robot.robot_KM.qr_dot * self.robot.dt  # In radians
        return self.robot.robot_KM.qr.astype(np.float64), qr_ddot.astype(np.float64)
        