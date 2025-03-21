import os
import sys
import numpy as np
import numpy.linalg as LA
np.set_printoptions(suppress=True)
from sympy import *

from .kinematic_controllers import Kinematic_Control
from .mpc_controller import Linear_MPC_JointSpace, Linear_MPC_TaskSpace
from .dmp import DMP
from .gaussianprocess import MIMOGaussianProcess

class Controller:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.KC = Kinematic_Control(self.robot)

    def start_jointspace_mpc(self, x_ref, N, state_cost, input_cost):
        self.jointspace_mpc_controller = Linear_MPC_JointSpace(self.robot.n, x_ref, N, state_cost, input_cost)
        q_min = self.robot.joint_limits["lower"]
        q_max = self.robot.joint_limits["upper"]
        q_dot = self.robot.joint_limits["vel_max"]
        self.jointspace_mpc_controller.set_joint_limits(q_min, q_max, q_dot)

    def start_taskspace_mpc(self, x_ref, N, state_cost, input_cost):
        self.taskspace_mpc_controller = Linear_MPC_TaskSpace(self.robot.n, x_ref, N, state_cost, input_cost)
        q_min = self.robot.joint_limits["lower"]
        q_max = self.robot.joint_limits["upper"]
        q_dot = self.robot.joint_limits["vel_max"]
        self.taskspace_mpc_controller.set_joint_limits(q_min, q_max, q_dot)

    def start_dmp(self, no_of_DMPs, no_of_basis, run_time, K, alpha):
        self.dmp = DMP(no_of_DMPs=no_of_DMPs, no_of_basis_func=no_of_basis, T=run_time, K=K, 
                       dt=self.robot.dt, alpha=alpha, method=self.robot.method, obstacles=self.robot.obstacles)

    def start_gp(self, input_dim, output_dim):
        self.gp = MIMOGaussianProcess(input_dim, output_dim)

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
        q_ddot = qd_ddot[:, np.newaxis] - u

        # Computed Torque Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ q_ddot + C + G
        return tau
    
    def jointspace_mpc_ctc(self, q, q_dot, qd_ddot):
        # Get desired acceleration from MPC
        state_vec = np.hstack((q, q_dot)).reshape(-1,1)  # (2*n, 1)
        u = self.jointspace_mpc_controller.compute_control(state_vec)
        q_ddot = qd_ddot[:, np.newaxis] - u
        
        # Compute torque using CTC
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        tau = M @ q_ddot + C + G
        return tau
    
    def taskspace_mpc_ctc(self, X, X_dot, q, q_dot):
        pass

    # WITHOUT contact force feedback
    def impedence_control_static(self, q, q_dot, E, E_dot, Dd, Kd):
        """
        Impedance control is widely used in robotics, especially in applications involving physical interaction 
        with uncertain environments. It's ideal for scenarios where a robot needs to respond flexibly to forces 
        exerted on it. Classic impedance control tries to establish a mass-spring-damper relationship between 
        external forces and robot motion.
        """
        _,J,_ = self.robot.robot_KM.J(q)

        M, C_vec, _, G = self.robot.compute_dynamics(q, q_dot)
       
        # Classical Impedance Controller with No Inertia
        impedence_force = Dd @ E_dot + Kd @ E
        tau = G - J.T @ impedence_force   # Cartesian PD control with gravity cancellation
        return tau
    
    def impedence_control_TT_1(self, q, q_dot, E, E_dot, Xd_ddot, Dd, Kd):
        """Impedance control with weighted pseudo-inverse"""
        # Task Jacobian
        _,J,_ = self.robot.robot_KM.J(q)   # only linear velocity
        J_inv = J.T @ LA.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

        _,J_dot,_ = self.robot.robot_KM.J_dot(q, q_dot)
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        
        # Compute desired acceleration
        qd_ddot = J_inv @ (Xd_ddot - J_dot @ q_dot.reshape((self.robot.n, 1)))
        
        # Standard impedance control
        impedence_force = Dd @ E_dot + Kd @ E
        tau = M @ qd_ddot + C + G - J.T @ impedence_force
        return tau
    
    # Guarantee of asymptotic convergence to zero tracking error (on Xd(t)) when F=0 (no contact situation)
    def impedence_control_TT_2(self, q, q_dot, E, E_dot, Xd_dot, Xd_ddot, Dd, Kd):
        _,J,_ = self.robot.robot_KM.J(q)    # only linear velocity
        J_inv = J.T @ LA.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

        _,J_dot,_ = self.robot.robot_KM.J_dot(q, q_dot)

        M, _, C, G = self.robot.compute_dynamics(q, q_dot)    # here, C is matrix
        
        qd_dot = J_inv @ Xd_dot
        qd_ddot = J_inv @ (Xd_ddot - J_dot @ qd_dot)

        impedence_force = Dd @ E_dot + Kd @ E
        tau = M @ qd_ddot + C @ qd_dot + G - J.T @ impedence_force
        return tau
    
    # (Passive) Dynamical-System based Impedence Controller
    def passive_control(self):
        pass
    
    """Simplified Acceleration-based Control Variation 1 (With Null Space Pre-multiplication of M)"""
    def torque_control_1(self, q, q_dot, qr_ddot):
        u = qr_ddot[:,np.newaxis]

        # Control Law
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)

        """Eqn (38) from https://doi.org/10.1177/0278364908091463"""
        tau = M @ u + C + G
        return tau

    """https://studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/"""
    def torque_control_2(self, Xd_ddot, q, q_dot):
        _,J,_ = self.robot.robot_KM.J(q)    # only linear velocity
        
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        Mx = self.robot.Mx(M, J)

        tau = (J.T @ (Mx @ Xd_ddot)) + G
        return tau
    
    """https://studywolf.wordpress.com/2013/09/17/robot-control-5-controlling-in-the-null-space/"""
    def torque_control_3(self, Xd_ddot, q, q_dot, Kp):
        _,J,_ = self.robot.robot_KM.J(q)    # only linear velocity

        """Null space control"""
        ## Option 1: causing Torque chattering
        tau_null = -Kp @ (q - self.robot.robot_KM.q)[:, np.newaxis]  # secondary controller working to keep the arm near its joint angles default resting positions.

        ## Option 2: No Torque chattering
        # tau_null = self.robot.robot_KM.manipulability_Jacobian(q, J)   # Manipulability Jacobian matrix
        
        M, C, _, G = self.robot.compute_dynamics(q, q_dot)
        Mx = self.robot.Mx(M, J)  # (3,3)

        # dynamically consistent generalized inverse
        """https://www.roboticsproceedings.org/rss07/p31.pdf"""
        J_pinv_T = Mx @ J @ LA.inv(M)
        
        tau = (J.T @ (Mx @ Xd_ddot)) + G + ((np.eye(self.robot.n) - J.T @ J_pinv_T) @ tau_null)
        return tau


    