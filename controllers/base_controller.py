import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *

from .kinematic_controllers import Kinematic_Control
from .mpc_controller import Linear_MPC
from .dmp import DMP

class Controller:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.KC = Kinematic_Control(self.robot)

    def start_mpc(self, x_ref, dt, N, state_cost, input_cost):
        self.mpc_controller = Linear_MPC(self.robot.n, x_ref, dt, N, state_cost, input_cost)
        q_min = self.robot.joint_limits["lower"]
        q_max = self.robot.joint_limits["upper"]
        q_dot = self.robot.joint_limits["vel_max"]
        self.mpc_controller.set_joint_limits(q_min, q_max, q_dot)

    def start_dmp(self):
        self.dmp = DMP(no_of_DMPs=3, no_of_basis_func=40, K=1000.0, dt=self.robot.dt, alpha=3.0)

    def q_tilda(self, q, qd):
        """ Computes the shortest angular distance between current position (q) and desired position (qd),
            Uses modulo operation to wrap angles to stay within [-π, π] range,
            This ensures the robot always takes the shortest path between angles"""
        q_tilda = np.zeros(self.robot.n)
        for i in range(self.robot.n):
            q_tilda[i] = ((qd[i] - q[i] + np.pi) % (np.pi * 2)) - np.pi 
        return q_tilda

    def pd_ctc(self, q, q_dot, qd, qd_dot, qd_ddot, Kp, Kd):
        """
        θ: joint angle
        𝜔 = (dθ/dt): joint velocity
        𝛼 = (d𝜔/dt): joint acceleration

        Robot Dynamics: M(θ)*𝛼 + C(θ,𝜔) + G(θ) = 𝛕
        tracking error: e(t) = θ_d(t) - θ(t)
                        d[e(t)]/dt = 𝜔_d(t) - 𝜔(t) 
                        dd[e(t)]/ddt = 𝛼_d(t) - 𝛼(t) 

        To eliminate nonlinear term defined a feed-forward control in this case, CTC law: 𝛕 = M(θ)*𝛼_d + C(θ,𝜔) + G(θ)
        Now, just using a feed-forward controller is not enough, since we do not have an exact model of our robot.

        So, we introduce a feed-back control term u in the CTC law: 𝛕 = M(θ)*(𝛼_d + u) + C(θ,𝜔) + G(θ)
        here, u is the outer loop feedback control input.

        Substituting into the robot dynamics: M(θ)*𝛼 + C(θ,𝜔) + G(θ) = M(θ)*(𝛼_d + u) + C(θ,𝜔) + G(θ)
        
        finally got (2nd order linear DS): 𝛼 = 𝛼_d + u  => (𝛼 - 𝛼_d) = u
        
        Now, if we select a control u that stabilizes (𝛼 - 𝛼_d) = u, then e(t) goes to 0

        A simple Outer feedback law that stabilizes the system along θ_d(t) is: u = - Kp*e(t) - Kd*(de(t)/dt)

        Input to the system: 𝛕 = M(θ)*{𝛼_d(t) - Kp*e(t) - Kd*(de(t)/dt)} + C(θ,𝜔) + G(θ)
        """
        # e = self.q_tilda(q, qd)[:, np.newaxis]
        e = (qd - q)[:, np.newaxis]
        e_dot = (qd_dot - q_dot)[:, np.newaxis]

        # Feed-back PD-control input
        u = - Kp @ e - Kd @ e_dot

        # Computed Torque Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ (qd_ddot[:, np.newaxis] - u) + C + G
        return tau
    
    """Not Working Yet"""
    def mpc_ctc(self, q, q_dot, qd_ddot):
        # Get desired acceleration from MPC
        state_vec = np.hstack((q, q_dot)).reshape(-1,1)  # (2*n, 1)
        u = self.mpc_controller.compute_control(state_vec)
        
        # Compute torque using CTC
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ (qd_ddot[:, np.newaxis] - u) + C + G
        return tau

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
    
    def impedence_control_TT_1(self, q, q_dot, E, E_dot, Xd_ddot, Dd, Kd):
        """Impedance control with weighted pseudo-inverse"""
        # Task Jacobian
        J = self.robot.robot_KM.J(q)
        J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

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
        J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

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

    """Velocity-based Control with Joint Velocity Integration"""
    def velocity_based_torque_control_1(self, q, q_dot, qr, qr_dot, qr_ddot, Kd, Kp):
        # Define tracking error    
        e = (qr - q).reshape((self.robot.n, 1)) 
        e_dot = (qr_dot - q_dot).reshape((self.robot.n, 1)) 

        # Feed-back PD-control Input
        u = qr_ddot[:,np.newaxis]

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(qr, qr_dot)

        """Eqn (14) from https://doi.org/10.1177/0278364908091463"""
        tau = M @ u + C + G + Kd @ e_dot + Kp @ e
        return tau

    """Velocity-based Control with Joint Velocity Integration"""
    def velocity_based_torque_control_2(self, q, q_dot, qr_dot, qr_ddot, Kd):
        # Define tracking error    
        e_dot = (qr_dot - q_dot).reshape((self.robot.n, 1)) 

        # Feed-back PD-control Input
        u = qr_ddot[:,np.newaxis]

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)

        """Eqn (23) from https://doi.org/10.1177/0278364908091463"""
        tau = M @ u + C + G + Kd @ e_dot
        return tau
    
    def torque_control_1(self, q, q_dot, qr_ddot, Kd):
        J = self.robot.robot_KM.J(q)

        J_inv = J.T @ np.linalg.inv(J @ J.T)    # pseudoinverse of the Jacobian

        # Hessian Matrix
        H = self.robot.robot_KM.Hessian(q)

        # Manipulability Jacobian matrix
        J_m = self.robot.robot_KM.manipulability_Jacobian(q, J, H)

        """Eqn (46) from https://doi.org/10.1177/0278364908091463"""
        zeta = -(Kd @ q_dot[:, np.newaxis] + 30 * J_m)
        
        # Feed-back PD-control Input
        u = qr_ddot[:,np.newaxis]

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ u + C + G + (np.eye(self.robot.n) - J_inv @ J) @ zeta
        return tau
    
    """Simplified Acceleration-based Control Variation 1 (With Null Space Pre-multiplication of M)"""
    def torque_control_2(self, q, q_dot, qr_ddot):
        u = qr_ddot[:,np.newaxis]

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)

        """Eqn (38) from https://doi.org/10.1177/0278364908091463"""
        tau = M @ u + C + G
        return tau

    