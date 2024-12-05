import os
import sys
import numpy as np
from math import *


class Kinematic_Control:
    def __init__(self, robot_model):
        self.robot = robot_model

    def velocity_based_control(self, dt, q, Ex, Xd_dot, Kp_task_space):
        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J)

        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian

        # Xd_dot is the desired task space velocities, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        Xr_dot = Xd_dot + Kp_task_space @ Ex
        
        zeta = - (10 * J_m)

        qr_dot_t1 = J_inv @ Xr_dot + (np.eye(self.robot.n) - J_inv @ J) @ zeta   # In radians/s
        qr_dot_t1 = qr_dot_t1.reshape(-1)

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + qr_dot_t1 * dt  # In radians

        # Numerical Integration
        qr_ddot = np.zeros((self.robot.n))         # this is creating much smoother torque
        
        return self.robot.robot_KM.qr.astype(np.float64), qr_dot_t1.astype(np.float64), qr_ddot.astype(np.float64)
        
    def acceleration_based_control_1(self, dt, q, q_dot, Ex, Ex_dot, Xd_ddot, Kp_task_space, Kd_task_space):
        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Hessian Matrix
        H = self.robot.robot_KM.Hessian_matrix(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J, H)

        # jacobian_dot matrix
        J_dot = self.robot.robot_KM.J_dot(q, q_dot, H)

        J_inv = J.T @ np.linalg.inv(J @ J.T)    # pseudoinverse of the Jacobian

        # Xd_ddot is the desired task space acceleration, Xd_dot is the desired task space velocity, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        # Ex_dot = Xd_dot - Xe_dot is the task-space velocity error, where, Xe is the end-effector velocity
        Xr_ddot = Xd_ddot + Kd_task_space @ Ex_dot + Kp_task_space @ Ex 

        zeta = - (30 * J_m)

        qr_ddot = J_inv @ (Xr_ddot - (J_dot @ self.robot.robot_KM.qr_dot_t0.reshape((self.robot.n, 1)))) + (np.eye(self.robot.n) - J_inv @ J) @ zeta   # In radians/s^2
        qr_ddot = qr_ddot.reshape(-1)

        # Numerical Differentation
        self.robot.robot_KM.qr_dot_t0 = self.robot.robot_KM.qr_dot_t0 + qr_ddot * dt  # In radians/s

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + self.robot.robot_KM.qr_dot_t0 * dt  # In radians

        return self.robot.robot_KM.qr.astype(np.float64), self.robot.robot_KM.qr_dot_t0.astype(np.float64), qr_ddot.astype(np.float64)
        
    def acceleration_based_control_2(self, dt, q, q_dot, Ex, Ex_dot, Xd_ddot, Kp_task_space, Kd_task_space, kd_joint_space):
        # jacobian matrix
        J = self.robot.robot_KM.J(q)

        # Hessian Matrix
        H = self.robot.robot_KM.Hessian_matrix(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J, H)

        # jacobian_dot matrix
        J_dot = self.robot.robot_KM.J_dot(q, q_dot, H)

        J_inv = J.T @ np.linalg.inv(J @ J.T)    # pseudoinverse of the Jacobian

        # Xd_ddot is the desired task space acceleration, Xd_dot is the desired task space velocity, Xd is the desired task space position
        # Ex = Xd - Xe is the task-space position error, where, Xe is the end-effector position
        # Ex_dot = Xd_dot - Xe_dot is the task-space velocity error, where, Xe is the end-effector velocity
        Xr_ddot = Xd_ddot + Kd_task_space @ Ex_dot + Kp_task_space @ Ex 

        zeta = -(kd_joint_space @ q_dot.reshape((self.robot.n, 1)) + 30 * J_m)

        qr_ddot = J_inv @ (Xr_ddot - J_dot @ self.robot.robot_KM.qr_dot_t0.reshape((self.robot.n, 1))) + (np.eye(self.robot.n) - J_inv @ J) @ zeta   # In radians/s^2
        qr_ddot = qr_ddot.reshape(-1)

        # Numerical Differentation
        self.robot.robot_KM.qr_dot_t0 = self.robot.robot_KM.qr_dot_t0 + qr_ddot * dt  # In radians/s

        # Numerical Differentation
        self.robot.robot_KM.qr = self.robot.robot_KM.qr + self.robot.robot_KM.qr_dot_t0 * dt  # In radians

        return self.robot.robot_KM.qr.astype(np.float64), qr_ddot.astype(np.float64)
        