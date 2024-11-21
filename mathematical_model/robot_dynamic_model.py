import os
import sys
import numpy as np
import numpy.linalg as LA
np.set_printoptions(suppress=True)
from sympy import *

from dynamic_modelling_methods import Euler_Lagrange
from plotting import RobotPlotter
from .robot_kinematic_model import Robot_KM

# Add the parent directory 'PhD' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Robot_Dynamics:
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

        self.robot_KM = Robot_KM(self.n, self.alpha, self.a, self.d, self.d_nn)  # Numeric Kinematic Model

        robot_DM = Euler_Lagrange(self.n, self.CG, self.MOI, self.d_nn)  # Symbolic Dynamic Model
        
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
        M(theta) * theta_ddot + C(theta, theta_dot) + G(theta) = tau

        M(theta) * theta_ddot = [tau - C(theta, theta_dot) - G(theta)] 
        
        A * Y = b
        Y = A^(-1) * b  =>  np.linalg.solve(A, b) 
        """
        M, C, _, G = self.compute_dynamics(q, q_dot)
        
        if ext_force is None:
            b = tau - (C + G)
        else:
            J = self.robot_KM.J(q)
            if ext_force.ndim == 1:
                ext_force = ext_force.reshape((3,1))
            b = tau - (C + G - J.T @ ext_force)

        b = b.astype(np.float64)  # Ensure b is float64

        q_ddot = np.linalg.solve(M, b).reshape(-1)

        if forward_int is None:
            return q_dot, q_ddot
        
        elif forward_int == 'euler_forward':
            q_dot_new = q_dot + q_ddot * self.dt
            q_new = q + q_dot * self.dt

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
            print("wrong forward_intergral chosen!")

    def M_matrix_task_space(self, M_q, J):
        M_x = LA.inv(J @ LA.inv(M_q) @ J.T)
        return M_x

    def M_C_matrix_task_space(self, M_q, C_q, J, J_dot):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        M_x = LA.inv(J @ LA.inv(M_q) @ J.T)
        C_x = J_inv.T @ C_q @ J_inv - M_x @ J_dot @ J_inv
        return M_x, C_x 

    def M_G_matrix_task_space(self, M_q, G_q, J):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        M_x = LA.inv(J @ LA.inv(M_q) @ J.T)
        G_x = J_inv.T @ G_q
        return M_x, G_x

    def M_C_G_matrix_task_space(self, M_q, C_q, G_q, J, J_dot):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        M_x = LA.inv(J @ LA.inv(M_q) @ J.T)
        C_x = J_inv.T @ C_q @ J_inv - M_x @ J_dot @ J_inv
        G_x = J_inv.T @ G_q
        return M_x, C_x, G_x
    
    def free_fall(self, time_frame, q, q_dot, tau=None):
        # For plotting
        self.X_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.Y_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.Z_plot = np.zeros((len(time_frame)+1, self.n + 2))

        self.dt = time_frame[1] - time_frame[0]
                
        if tau is None:
            tau = np.zeros((self.n,1))

        for frame in range(time_frame.shape[0]-1):
            # Forward Kinematics
            X_cord, Y_cord, Z_cord = self.robot_KM.FK(q)
            
            # Store in memory
            self.memory(frame, X_cord, Y_cord, Z_cord) 

            # Update state using RK4
            q, q_dot, _ = self.forward_dynamics(q, q_dot, tau, forward_int='rk4')

        """Animation Setup"""
        plotter = RobotPlotter(self)
        anim = plotter.setup_free_fall_plot()
        plotter.show()

    def computed_torque_control(self, time_frame, q, q_dot, q_des, q_dot_des, q_ddot_des, Kp, Kd):
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
        
        # For plotting
        self.X_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.Y_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.Z_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.joint_error_plot = np.zeros((self.n, len(time_frame)+1))
        self.tau_plot = np.zeros((self.n, len(time_frame)+1))

        self.dt = time_frame[1] - time_frame[0]
        self.T = time_frame

        for t in range(self.T.shape[0]-1):
            # Forward Kinematics
            X_cord, Y_cord, Z_cord = self.robot_KM.FK(q)

            # Define tracking error
            if q.shape != q_des[:,[t]].shape:
                Q = q.reshape(q_des[:,[t]].shape)
            e = q_des[:,[t]] - Q
            
            if q_dot.shape != q_dot_des[:,[t]].shape:
                Q_dot = q_dot.reshape(q_dot_des[:,[t]].shape)
            e_dot = q_dot_des[:,[t]] - Q_dot

            # Feed-back PD-control Input
            u = (Kp @ e) + (Kd @ e_dot)

            # Computer Torque Control Law
            M, C, _, G = self.compute_dynamics(q, q_dot)
            tau = M @ (q_ddot_des[:,[t]] + u) + C + G

            # Store in memory
            self.memory(t, X_cord, Y_cord, Z_cord, e, tau) 

            # Robot next state based on control input
            # q, q_dot = self.rk4_step(q, q_dot, tau)
            q, q_dot, _ = self.forward_dynamics(q, q_dot, tau, forward_int='rk4')

        """Animation Setup"""
        plotter = RobotPlotter(self)
        anim = plotter.setup_computed_torque_in_joint_space()
        plotter.show()

    # WITHOUT contact force feedback
    def impedence_control_static(self, q, q_dot, E, E_dot, Dd, Kd):
        """
        Impedance control is widely used in robotics, especially in applications involving physical interaction 
        with uncertain environments. It's ideal for scenarios where a robot needs to respond flexibly to forces 
        exerted on it. Classic impedance control tries to establish a mass-spring-damper relationship between 
        external forces and robot motion.
        """
        J = self.robot_KM.J(q)

        M, C, _, G = self.compute_dynamics(q, q_dot)
       
        # Classical Impedance Controller with No Inertia
        impedence_force = Dd @ E_dot + Kd @ E
        tau = G - J.T @ impedence_force   # Cartesian PD control with gravity cancellation
        return tau

    def _get_joint_weights(self, q):
        """Calculate weights based on distance from joint limits"""
        weights = np.ones(self.n)
        for i in range(self.n):
            q_max = self.joint_limits['upper'][i]
            q_min = self.joint_limits['lower'][i]
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
        J = self.robot_KM.J(q)
        J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

        # # Get joint weights
        # W = np.diag(self._get_joint_weights(q))
        
        # # Weighted damped pseudo-inverse
        # J_inv = np.linalg.inv(W) @ J.T @ np.linalg.inv(J @ np.linalg.inv(W) @ J.T + 0.005 * np.eye(J.shape[0]))
        
        J_dot = self.robot_KM.J_dot(q, q_dot)
        M, C, _, G = self.compute_dynamics(q, q_dot)
        
        # Compute desired acceleration
        qd_ddot = J_inv @ (Xd_ddot - J_dot @ q_dot.reshape((self.n, 1)))
        
        # Standard impedance control
        impedence_force = Dd @ E_dot + Kd @ E
        tau = M @ qd_ddot + C + G - J.T @ impedence_force
        return tau
    
    # Guarantee of asymptotic convergence to zero tracking error (on Xd(t)) when F=0 (no contact situation)
    def impedence_control_TT_2(self, q, q_dot, E, E_dot, Xd_dot, Xd_ddot, Dd, Kd):
        J = self.robot_KM.J(q)
        # J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(J.shape[0]))    # pseudoinverse of the Jacobian

        # Get joint weights
        W = np.diag(self._get_joint_weights(q))
        
        # Weighted damped pseudo-inverse
        J_inv = np.linalg.inv(W) @ J.T @ np.linalg.inv(J @ np.linalg.inv(W) @ J.T + 0.005 * np.eye(J.shape[0]))

        J_dot = self.robot_KM.J_dot(q, q_dot)

        M, _, C, G = self.compute_dynamics(q, q_dot)    # here, C is matrix
        
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
        e_dot = (qr_dot - q_dot).reshape((self.n, 1)) 

        # Feed-back PID-control Input
        u = qr_ddot.reshape((self.n,1)) + (Kd @ e_dot)

        # Control Law
        M, C, _, G = self.compute_dynamics(q, q_dot)
        tau = M @ u + C + G
        return tau
    
    def torque_control_2(self, q, q_dot, qr_ddot):
        u = qr_ddot.reshape((self.n,1))

        # Control Law
        M, C, _, G = self.compute_dynamics(q, q_dot)
        tau = M @ u + C + G
        return tau
    
    def show_plot(self):
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
        self.dt = dt
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