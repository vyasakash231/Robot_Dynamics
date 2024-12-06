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

    def ctc(self, q, q_dot, qd, qd_dot, qd_ddot, Kp, Kd):
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

        A simple Outer feedback law that stabilizes the system along θ_d(t) is: u = - Kp*e(t) - Kd*(d[e(t)]/dt)

        Input to the system: 𝛕 = M(θ)*{𝛼_d(t) - Kp*e(t) - Kd*(d[e(t)]/dt)} + C(θ,𝜔) + G(θ)
        """
        if q.shape != qd.shape:
            q = q.reshape(qd.shape)
        e = qd - q

        if q_dot.shape != qd_dot.shape:
            q_dot = q_dot.reshape(qd_dot.shape)
        e_dot = qd_dot - q_dot

        # Feed-back PD-control input
        u = Kp @ e + Kd @ e_dot

        # Computed Torque Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ (qd_ddot + u) + C + G
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

    def torque_control_1(self, q, q_dot, qr_dot, qr_ddot, Kd):
        # Define tracking error    
        e_dot = (qr_dot - q_dot).reshape((self.robot.n, 1)) 

        # Feed-back PD-control Input
        u = qr_ddot[:,np.newaxis] + (Kd @ e_dot)

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ u + C + G
        return tau
    
    def torque_control_2(self, q, q_dot, qr_ddot):
        u = qr_ddot.reshape[:,np.newaxis]

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ u + C + G
        return tau
    
    def stable_PD(self, q=None, q_dot=None, qd=[], qd_dot=[], Kp=[], Kd=[]):
        if q is None:
            q = self.robot.robot_KM.q
        if q_dot is None:
            q_dot = np.zeros(self.robot.n)
        
        q_error = (qd - q)[:, np.newaxis]
        q_dot_error = (qd_dot - q_dot)[:, np.newaxis]
        
        # Feed-Back PD control
        u = Kp @ q_error + Kd @ q_dot_error 
        
        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        b =  u - (C + G)

        # Forward Dynamics
        q_ddot = np.linalg.solve(M, b)

        tau = u - Kd @ q_ddot * self.robot.dt
        return tau