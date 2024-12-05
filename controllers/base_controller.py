import os
import sys
import numpy as np
import numpy.linalg as LA
np.set_printoptions(suppress=True)
from sympy import *

from .kinematic_controllers import Kinematic_Control


class Controller(Kinematic_Control):
    def __init__(self, robot_model):
        super().__init__(robot_model)
        self.robot = robot_model

    # WITHOUT contact force feedback
    def impedence_control_static(self, q, q_dot, E, E_dot, Dd, Kd):
        """
        Impedance control is widely used in robotics, especially in applications involving physical interaction 
        with uncertain environments. It's ideal for scenarios where a robot needs to respond flexibly to forces 
        exerted on it. Classic impedance control tries to establish a mass-spring-damper relationship between 
        external forces and robot motion.
        """
        J = self.robot.robot_KM.J(q)

        M, C_vec, _, G = self.robot.compute_dynamics(q, q_dot)
       
        # Classical Impedance Controller with No Inertia
        impedence_force = Dd @ E_dot + Kd @ E
        tau = G - J.T @ impedence_force   # Cartesian PD control with gravity cancellation
        return tau

    def _get_joint_weights(self, q):
        """Calculate weights based on distance from joint limits"""
        weights = np.ones(self.robot.n)
        for i in range(self.robot.n):
            q_max = self.robot.joint_limits['upper'][i]
            q_min = self.robot.joint_limits['lower'][i]
            q_mid = (q_max + q_min) / 2
            q_range = (q_max - q_min) / 2
            
            # Normalized distance from middle (-1 to 1)
            dist = (q[i] - q_mid) / q_range
            
            # Increase weight when near limits
            if abs(dist) > 0.75:  # Start increasing weight at 85% of range
                weights[i] = 1.0 + 10.0 * (abs(dist) - 0.75)**2
        return weights
    
    def impedence_control_TT_1(self, q, q_dot, E, E_dot, Xd_ddot, Dd, Kd):
        """Impedance control with weighted pseudo-inverse"""
        # Task Jacobian
        J = self.robot.robot_KM.J(q)
        J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

        # # Get joint weights
        # W = np.diag(self._get_joint_weights(q))
        
        # # Weighted damped pseudo-inverse
        # J_inv = np.linalg.inv(W) @ J.T @ np.linalg.inv(J @ np.linalg.inv(W) @ J.T + 0.005 * np.eye(J.shape[0]))
        
        J_dot = self.robot.robot_KM.J_dot(q, q_dot)
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        
        # Compute desired acceleration
        qd_ddot = J_inv @ (Xd_ddot - J_dot @ q_dot.reshape((self.robot.n, 1)))
        
        # Standard impedance control
        impedence_force = Dd @ E_dot + Kd @ E
        tau = M @ qd_ddot + C + G - J.T @ impedence_force
        return tau
    
    # Guarantee of asymptotic convergence to zero tracking error (on Xd(t)) when F=0 (no contact situation)
    def impedence_control_TT_2(self, q, q_dot, E, E_dot, Xd_dot, Xd_ddot, Dd, Kd):
        J = self.robot.robot_KM.J(q)
        # J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

        # Get joint weights
        W = np.diag(self._get_joint_weights(q))
        
        # Weighted damped pseudo-inverse
        J_inv = np.linalg.inv(W) @ J.T @ np.linalg.inv(J @ np.linalg.inv(W) @ J.T + 0.005 * np.eye(J.shape[0]))

        J_dot = self.robot.robot_KM.J_dot(q, q_dot)

        M, _, C, G = self.robot.compute_dynamics(q, q_dot)    # here, C is matrix
        
        qd_dot = J_inv @ Xd_dot
        qd_ddot = J_inv @ (Xd_ddot - J_dot @ qd_dot)

        impedence_force = Dd @ E_dot + Kd @ E
        tau = M @ qd_ddot + C @ qd_dot + G - J.T @ impedence_force
        return tau
    
    # (Passive) Dynamical-System based Impedence Controller
    def passive_control(self):
        pass

    def torque_control_1(self, q, q_dot, qr_dot, qr_ddot, Kd):
        # Define tracking error    
        e_dot = (qr_dot - q_dot).reshape((self.robot.n, 1)) 

        # Feed-back PID-control Input
        u = qr_ddot.reshape((self.robot.n,1)) + (Kd @ e_dot)

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ u + C + G
        return tau
    
    def torque_control_2(self, q, q_dot, qr_ddot):
        u = qr_ddot.reshape((self.robot.n,1))

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ u + C + G
        return tau