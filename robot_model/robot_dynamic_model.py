import os
import sys
import numpy as np
import numpy.linalg as LA
np.set_printoptions(suppress=True)
from sympy import *

from dynamic_modelling_methods import Euler_Lagrange, RNEA
from plotting import RobotPlotter
from .robot_kinematic_model import Robot_KM
from .utils import *


class Robot_Dynamics:
    # TODO: Forward Dynamics: Brute Force (Matrix Inversion), Articulated-Body Algorithm
    # TODO: Inverse Dynamics: Recursive Newton-Euler Algorithm
    # TODO: Mass Matrix: Composite Rigid Body Algorithm 
    
    def __init__(self, kinematic_property={}, mass=[], COG_wrt_body=[], MOI_about_body_COG=[], joint_limits={}, file_name=None):
        self.n = kinematic_property['dof']
        self.alpha = kinematic_property['alpha']
        self.a = kinematic_property['a']
        self.d = kinematic_property['d']
        self.d_nn = kinematic_property['d_nn']

        self.m = mass
        self.CG = COG_wrt_body  # location of COG wrt to DH-frame / body frame
        if len(MOI_about_body_COG) == 0:  # if list is empty then MOI is zero
            self.MOI = [np.array([[0,0,0],[0,0,0],[0,0,0]]) for _ in range(len(self.m))]
        else:  # else MOI is non-zero 
            self.MOI = MOI_about_body_COG  # MOI of the link about COG based on DH-frame / body frame

        # Add joint limits
        self.joint_limits = joint_limits

        self.robot_KM = Robot_KM(self.n, self.alpha, self.a, self.d, self.d_nn, joint_limits)  # Numeric Kinematic Model

        # robot_DM = Euler_Lagrange(self.n, self.CG, self.MOI, self.d_nn)  # Symbolic Dynamic Model using Euler-Lagrange Method
        robot_DM = RNEA(self.n, self.CG, self.MOI, self.d_nn)  # Symbolic Dynamic Model using Recursive Netwon-Euler Method
        
        # Check if the file already exists
        if os.path.exists('../models/'+file_name+'.pkl'):
            print(f"{file_name} already exists. Skipping recreation.")
        else:
            # Code to create the file and save the data
            M_sym, C_vec_sym, C_mat_sym, G_sym = robot_DM.mcg_matrix(self.alpha, self.a, self.d, self.m)  # Symbolic Matrices
            robot_DM.save_equations(M_sym, C_vec_sym, C_mat_sym, G_sym, file_name)  # Save Matrices
            print(f"{file_name} has been created.")

        # Load the pre-computed symbolic equations
        self.equations = robot_DM.load_equations(file_name)

    def compute_dynamics(self, q, q_dot):
        M_args = q
        C_args = np.concatenate((q, q_dot))
        G_args = q

        M = self.equations['M_lambda'](*M_args)
        C_vec = self.equations['C_vec_lambda'](*C_args)
        C_mat = self.equations['C_mat_lambda'](*C_args)
        G = self.equations['G_lambda'](*G_args)
        return M, C_vec, C_mat, G
    
    def memory(self, X_cord, Y_cord, Z_cord, Er=None, tau=None, Xd=None, Ex=None, Ex_dot=None, wrench=None):
        self.X_plot.append(X_cord)
        self.Y_plot.append(Y_cord)
        self.Z_plot.append(Z_cord)

        if Er is not None:
            self.joint_error_plot.append(Er.reshape(-1))
        if tau is not None:
            self.tau_plot.append(tau.reshape(-1))
        if Xd is not None:
            self.Xd_plot.append(Xd.reshape(-1))
        if Ex is not None:
            self.position_error_plot.append(Ex.reshape(-1))
        if Ex_dot is not None:
            self.velocity_error_plot.append(Ex_dot.reshape(-1))
        if wrench is not None:
            self.wrench.append(wrench.reshape(-1))

    def forward_dynamics(self, q, q_dot, tau, forward_int=None, ext_force=None):
        """ 
        Using Brute Force Approach: 
        M(theta) * theta_ddot + C(theta, theta_dot) + G(theta) = tau
        M(theta) * theta_ddot = [tau - C(theta, theta_dot) - G(theta)] 
        
        A * Y = b
        Y = A^(-1) * b  =>  np.linalg.solve(A, b)
        
        * Finding A^(-1) has O(n^3) time complexity using gaussian elimination, but we can use it till (n <= 6)
        * For, (n > 6) we have to use Articulated Body Algorithm, it's time complexity is O(n) 
        """
        M, C_vec, _, G = self.compute_dynamics(q, q_dot)  
            
        if ext_force is None:
            b = tau - (C_vec + G)
        else:
            J = self.robot_KM.J(q)
            if ext_force.ndim == 1:
                ext_force = ext_force.reshape((3,1))
            b = tau - (C_vec + G - J.T @ ext_force)

        b = b.astype(np.float64)  # Ensure b is float64
        q_ddot = np.linalg.solve(M, b).reshape(-1)

        if forward_int is None:
            return q_dot, q_ddot
        
        elif forward_int == 'euler_forward':
            q_dot_new = q_dot + q_ddot * self.dt
            q_new = q + q_dot_new * self.dt
            return q_new, q_dot_new, q_ddot
        
        elif forward_int == 'rk4':
            """
            θ: joint angle
            𝜔 = (dθ/dt): joint velocity
            𝛼 = (d𝜔/dt): joint acceleration

            0) Second-order Dynamical System: 𝛼 = f(θ, 𝜔) => theta_ddot = f(theta, theta_dot)
           
                                     |x1|   |  theta  |   |θ|
            1) state-space form: X = |  | = |         | = | |
                                     |x2|   |theta_dot|   |𝜔|

                                                |x1_dot|   |theta_dot |   |𝜔|   |   x2   |
            2) taking time derivative: X_dot  = |      | = |          | = | | = |        | = g(x1,x2) = g(X)
                                                |x2_dot|   |theta_ddot|   |𝛼|   |f(x1,x2)| 
                                
            3) Runga-Kutta 4th order:
                               |  k1_q  |        |   x2   |
            k1 = dt * g(X) =>  |        | = dt * |        |
                               |k1_q_dot|        |f(x1,x2)| 

                                     |  k2_q  |        |        x2 + k1_q_dot/2        |
            k2 = dt * g(X + k1/2) => |        | = dt * |                               |
                                     |k2_q_dot|        |f(x1 + k1_q/2, x2 + k1_q_dot/2)|                                                      
                                               
                                     |  k3_q  |        |        x2 + k2_q_dot/2        |
            k3 = dt * g(X + k2/2) => |        | = dt * |                               |
                                     |k3_q_dot|        |f(x1 + k2_q/2, x2 + k2_q_dot/2)|  
                                               
                                     |  k4_q  |        |        x2 + k3_q_dot      |
            k4 = dt * g(X + k3/2) => |        | = dt * |                           |
                                     |k4_q_dot|        |f(x1 + k3_q, x2 + k3_q_dot)|  

            4) compute next-state vector: X(t+1) = X(t) + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

            |x1(t+1)|   |x1(t)|            |      k1_q + 2*k2_q + 2*k3_q + k4_q          |
            |       | = |     | + (dt/6) * |                                             |
            |x2(t+1)|   |x2(t)|            |k1_q_dot + 2*k2_q_dot + 2*k3_q_dot + k4_q_dot|
            """
            h = self.dt
            k1_q, k1_q_dot = self.forward_dynamics(q, q_dot, tau)
            k2_q, k2_q_dot = self.forward_dynamics(q + 0.5*h*k1_q, q_dot + 0.5*h*k1_q_dot, tau)
            k3_q, k3_q_dot = self.forward_dynamics(q + 0.5*h*k2_q, q_dot + 0.5*h*k2_q_dot, tau)
            k4_q, k4_q_dot = self.forward_dynamics(q + h*k3_q, q_dot + h*k3_q_dot, tau)
            
            q_new = q + (h/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
            q_dot_new = q_dot + (h/6) * (k1_q_dot + 2*k2_q_dot + 2*k3_q_dot + k4_q_dot)
            q_ddot = k1_q_dot
            return q_new, q_dot_new, q_ddot   
            
        else: 
            raise ValueError("wrong forward intergral chosen!")

    """Eqn (18) from https://doi.org/10.1109/JRA.1987.1087068"""
    def Mx(self, Mq, J):
        threshold = 1e-3

        Mx_inv = J @ LA.inv(Mq) @ J.T
        if abs(np.linalg.det(Mx_inv)) >= threshold:
            Mx = LA.inv(Mx_inv)
        else:
            Mx = LA.pinv(Mx_inv, rcond=threshold * 0.1)
        return Mx

    def Cx(self, Mq, Cq, J, J_dot):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        Mx = self.Mx(Mq, J)
        Cx = J_inv.T @ Cq @ J_inv - Mx @ J_dot @ J_inv
        return Cx 

    """Eqn (25) from https://doi.org/10.1109/JRA.1987.1087068"""
    def Gx(self, Gq, J):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        Gx = J_inv.T @ Gq
        return Gx

    def compute_MCGx(self, Mq, Cq, Gq, J, J_dot):
        Mx = self.Mx(Mq, J)
        Cx = self.Cx(Mq, Cq, J, J_dot)
        Gx = self.CGx(Gq, J)
        return Mx, Cx, Gx
    
    def free_fall(self, q, q_dot, tau=None):
        if tau is None:
            tau = np.zeros((self.n,1))
        
        for _ in range(self.T.shape[0]-1):
            # Forward Kinematics
            X_cord, Y_cord, Z_cord = self.robot_KM.EE(q)
            
            # Store in memory
            self.memory(X_cord, Y_cord, Z_cord) 
            
            # Update state using RK4
            q, q_dot, _ = self.forward_dynamics(q, q_dot, tau, forward_int='rk4')

        """Animation Setup"""
        plotter = RobotPlotter(self)
        anim = plotter.setup_free_fall_plot()
        plotter.show()

    def show_plot_tauj(self):
        """Animation Setup"""
        plotter = RobotPlotter(self)
        anim = plotter.setup_computed_torque_in_joint_space()
        plotter.show()
    
    def show_plot_taux(self):
        """Animation Setup"""
        plotter = RobotPlotter(self)
        anim = plotter.setup_computed_torque_in_task_space()
        plotter.show()

    def show_plot_impedence(self):
        """Animation Setup"""
        plotter = RobotPlotter(self)
        anim = plotter.setup_impedence_control_in_task_space()
        plotter.show()
    
    def plot_start(self, dt=None, time_frame=None):
        self.robot_KM.dt = self.dt = dt
        self.T = None

        if time_frame is not None:
            self.T = time_frame

        # For plotting
        self.X_plot = []
        self.Y_plot = []
        self.Z_plot = []
        self.joint_error_plot = []
        self.position_error_plot = []
        self.velocity_error_plot = []
        self.tau_plot = []
        self.Xd_plot = []
        self.wrench = []