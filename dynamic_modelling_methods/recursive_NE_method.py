from math import *
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .homo_transformation import *


class RNEA:
    g = symbols("g")  # gravity
    def __init__(self, n, location_COM_wrt_body, MOI_wrt_body_CG, d_nn, time_frame):
        self.n = n
        self.alpha = symarray('alpha',n)
        self.a = symarray('a',n)
        self.d = symarray('d',n)
        self.theta_vec = symarray('theta',n)
        self.d_nn = d_nn

        # For plotting
        self.X_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.Y_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.Z_plot = np.zeros((len(time_frame)+1, self.n + 2))
        self.dt = time_frame[1] - time_frame[0]

        self.mass_vec = symarray('m',n)  # mass of each link
        self.G_vec = Matrix([[0],[RNEA.g],[0]])  # Gravity Vector
        self.theta_dot_vec = symarray('theta_dot',self.n)  # shape -> (n,)
        self.theta_ddot_vec = symarray('theta_ddot',self.n)  # shape -> (n,)

        self.omega_vec = [zeros(3,1) for _ in range(self.n+1)]  # Angular velocity of links wrt to ground frame 
        self.omega_dot_vec = [zeros(3,1) for _ in range(self.n+1)]  # Angular acceleration of ground link wrt to ground frame 
        self.O_ddot_vec = [zeros(3,1) for _ in range(self.n+1)]  # Linear acceleration of the Joint/body frame wrt base frame

        self.C_ddot_vec = [zeros(3,1) for _ in range(self.n)]  # Linear acceleration of link COM

        # coordinates of COM of link wrt to body frame
        self.b_ii = location_COM_wrt_body 

        # Import manipulator Transformation data using DH parameters (a_n_n -> coord of (i)th joint wrt (i-1)th joint)
        self.Rot_mat, self.O, self.a_i0 = symbolic_transformation_matrix(self.n,self.alpha,self.a,self.d,self.theta_vec,self.d_nn)
        
        # z-vector (last column) in rotation matrix 
        self.Z_n_0 = []  # (3,n)
        for R in self.Rot_mat:
            self.Z_n_0.append(R[:,[-1]])
        
        self.I_i_i = MOI_wrt_body_CG  # link MOI wrt body CG frame (as list of matrices)
        self.I_i_0 = []  # MOI of each link wrt base frame
        for i in range(self.n):
            self.I_i_0.append(self.Rot_mat[i] * self.I_i_i[i] * self.Rot_mat[i].T)  # I = R * I * R.T
         
    def _forward_pass(self):
        self.b_i0 = zeros(3,self.n+1)
        self.f_i_0 = [zeros(3,1) for _ in range(self.n)]  # Net-force of (i)th link COM wrt to base frame
        self.n_i_0 = [zeros(3,1) for _ in range(self.n)]  # Net-moment of (i)th link wrt to base frame
        self.r_i0 = zeros(3,self.n+1)

        for i in range(self.n):
            self.omega_vec[i+1] = self.omega_vec[i] + self.Z_n_0[i] * self.theta_dot_vec[i]  # Angular velocity of links wrt to ground frame
            
            self.b_i0[:,i+1] = self.Rot_mat[i] * self.b_ii[i]
            
            ext_1 = self.Z_n_0[i] * self.theta_dot_vec[i]
            self.omega_dot_vec[i+1] = self.omega_dot_vec[i] + self.omega_vec[i+1].cross(ext_1) + self.Z_n_0[i] * self.theta_ddot_vec[i]  # Angular acceleration
            
            ext_2 = self.O[:,i+1] - self.O[:,i]
            ext_3 = self.omega_vec[i].cross(ext_2)
            self.O_ddot_vec[i+1] = self.O_ddot_vec[i] + self.omega_dot_vec[i].cross(ext_2) + self.omega_vec[i].cross(ext_3)  # Linear acceleration of joints
            
            ext_4 = self.omega_vec[i+1].cross(self.b_i0[:,i])
            self.C_ddot_vec[i] = self.O_ddot_vec[i+1] + self.omega_dot_vec[i+1].cross(self.b_i0[:,i]) + self.omega_vec[i+1].cross(ext_4) # Linear acceleration of COM
            
            '''D-Alembert demonstrated that by adding “inertial force” and “inertial torque” or moment, 
            an accelerating rigid body can be transformed into an equal static system. 
            The inertial torque can act anywhere and the inertial force must act through the centre of mass.'''
            self.f_i_0[i] = self.mass_vec[i] * self.C_ddot_vec[i]  # Inertial force acting on COM (if we consider D’Alembert’s Principle)
            
            self.n_i_0[i] = self.I_i_0[i] * self.omega_dot_vec[i+1] + self.omega_vec[i+1].cross(self.I_i_0[i] * self.omega_vec[i+1])  # Inertial moment / torque on each joint
            
            self.r_i0[:,i+1] = self.a_i0[:,i+1] - self.b_i0[:,i+1]  # r_i0 = r_{i}_0

    def _backward_pass(self):
        self.f_j_0 = [zeros(3,1) for _ in range(self.n+1)]  # Net-force applied by (n-1)th link on last / (n)th link 
        self.n_j_0 = [zeros(3,1) for _ in range(self.n+1)]  # Net-moment applied by (n-1)th link on last / (n)th link 
        self.tau = zeros(self.n,1)   # Driving Torque
        
        for j in range(self.n-1, -1, -1):  # [n, n-1, n-2, n-3, ......, 0]
            self.f_j_0[j] = self.f_i_0[j] + self.f_j_0[j+1] - (self.mass_vec[j] * self.G_vec)
            self.n_j_0[j] = self.n_i_0[j] + self.n_j_0[j+1] + self.b_i0[:,j+1].cross(self.f_j_0[j]) + self.r_i0[:,j+1].cross(self.f_j_0[j+1])
            self.tau[j] = self.Z_n_0[j].T * self.n_j_0[j]  # Driving Torque

    def mcg_matrix(self, alpha, a, d, m):
        self.numeric_DH = [alpha, a, d]

        self._forward_pass()  # Perform forward pass
        self._backward_pass()  # Perform backward pass

        # substitute DH-parameter and mass in each equation
        tau = zeros(self.n,1)   # Driving Torque
        for i in range(self.n):
            tau[i] = self.tau[i].subs([(k, alpha[idx]) for idx,k in enumerate(self.alpha)] + \
                                           [(k, a[idx]) for idx,k in enumerate(self.a)] + \
                                           [(k, d[idx]) for idx,k in enumerate(self.d)] + \
                                           [(k, m[idx]) for idx,k in enumerate(self.mass_vec)])

        tau_C = tau.copy()
        tau_M = tau.copy()
        tau_G = tau.copy()

        # C matrix formulation
        self.C = tau_C.subs([(k,0) for k in self.theta_ddot_vec] + [(RNEA.g, 0)])  # substitute 0 for k --> (k,0)
        
        # M matrix formulation
        self.M = symarray('M',(self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                if j == 0:
                    theta_ddot = Array(self.theta_ddot_vec[j+1:])
                elif j == self.n-1:
                    theta_ddot = Array(self.theta_ddot_vec[:j])
                else:
                    theta_ddot = Array(Array(self.theta_ddot_vec[:j]).tolist() + Array(self.theta_ddot_vec[j+1:]).tolist()) 
               
                M = tau_M[i].subs([(self.theta_ddot_vec[j], 1)] + [(k,0) for k in self.theta_dot_vec] + [(RNEA.g, 0)])
                self.M[i,j] = M.subs([(k,0) for k in theta_ddot])  
                
        # G vector formulation
        self.G = tau_G.subs([(k,0) for k in self.theta_ddot_vec] + [(k,0) for k in self.theta_dot_vec] + [(RNEA.g, -9.81)])
        
        return simplify(self.M), simplify(self.C), simplify(self.G)  # symbolic
    
    def matrix_M(self, q):
        """ M(theta) """
        # If self.M is already a numpy array, convert it back to SymPy Matrix
        if isinstance(self.M, np.ndarray):
            self.M = Matrix(self.M)

        # Create a list of (symbol, value) pairs for substitution
        subs_list = [(sym, val) for sym, val in zip(self.theta_vec, q)]
    
        M_substituted = self.M.subs(subs_list)  # Substitute values into symbolic matrix
        return np.array(M_substituted, dtype=float)  # Convert to numpy array

    def matrix_C(self, q, q_dot):
        """ C(theta, theta_dot) """
        # Create substitution lists for both q and q_dot
        subs_list = [(sym, val) for sym, val in zip(self.theta_vec, q)]
        subs_list.extend([(sym, val) for sym, val in zip(self.theta_dot_vec, q_dot)])
        
        C_substituted = self.C.subs(subs_list)  # Substitute values into symbolic matrix
        return np.array(C_substituted, dtype=float)  # Convert to numpy array

    def matrix_G(self, q):
        """ G(theta) """
        # Create a list of (symbol, value) pairs for substitution
        subs_list = [(sym, val) for sym, val in zip(self.theta_vec, q)]

        G_substituted = self.G.subs(subs_list)  # Substitute values into symbolic matrix
        return np.array(G_substituted, dtype=float)  # Convert to numpy array
    
    def memory(self,frame,X_cord,Y_cord,Z_cord):
        self.X_plot[frame] = X_cord
        self.Y_plot[frame] = Y_cord
        self.Z_plot[frame] = Z_cord 
    
    """ This approach for using RNE in forward dynamics is often called the {Composite-Rigid-Body Algorithm} """
    def forward_dynamics(self, initial_state, t, tau=None, contact_force=None):
        q = np.array(initial_state[0::2]) 
        q_dot = np.array(initial_state[1::2])

        frame = 0

        alpha, a, d = self.numeric_DH
        
        # Forward Kinematics
        X_cord, Y_cord, Z_cord = forward_kinematics(self.n, alpha, a, d, q, self.d_nn)
        
        # Store in memory
        self.memory(frame, X_cord, Y_cord, Z_cord) 
                
        if tau is None:
            tau = np.zeros((n,1))

        if contact_force is None:
            contact_force = np.zeros((n,1))

        for _ in range(len(t)):
            M = self.matrix_M(q)
            C = self.matrix_C(q, q_dot)
            G = self.matrix_G(q)

            # Compute accelerations
            """ 
            M(theta) * theta_ddot + C(theta, theta_dot) + G(theta) = tau

            M(theta) * theta_ddot = [tau - C(theta, theta_dot) - G(theta)] 
            
            A * Y = b
            Y = A^(-1) * b  =>  np.linalg.solve(A, b) 
            """
            b = tau + contact_force - C - G
            print(M,b)
            q_ddot = np.linalg.solve(M, b)  # shape (n,1) 
            q_dot = q_dot + q_ddot.reshape(-1) * self.dt   # shape (n,)
            q = q + q_dot * self.dt   # shape (n,)
            
            # Forward Kinematics
            X_cord, Y_cord, Z_cord = forward_kinematics(self.n, alpha, a, d, q, self.d_nn)
            
            # Store in memory
            frame += 1
            self.memory(frame, X_cord, Y_cord, Z_cord) 

        """Animation Setup"""
        fig = plt.figure(figsize=(12, 10))
        self.ax = fig.add_subplot(111, projection='3d')

        # Create the animation
        anim = FuncAnimation(fig, self.update, frames=self.X_plot.shape[0], interval=30, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()
        
    def update(self, frame):
        self.ax.clear()
        k = frame
        
        for j in range(self.X_plot.shape[1]):
            self.ax.plot(self.X_plot[k,j:j+2], self.Y_plot[k,j:j+2], self.Z_plot[k,j:j+2], '-', linewidth=9)
            self.ax.plot(self.X_plot[k,j], self.Y_plot[k,j], self.Z_plot[k,j], 'ko', linewidth=10)

        # trajectory of EE
        self.ax.plot(self.X_plot[:k+1, -1], self.Y_plot[:k+1, -1], self.Z_plot[:k+1, -1], linewidth=1.25, color='m') 

        self.ax.view_init(elev=90, azim=-90)

        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_zlim(-0.25, 0.25)
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        return self.ax


if __name__ == "__main__":
    # l1, l2, m1, m2 = Symbol("l1"), Symbol("l2"), Symbol("m1"), Symbol("m2")

    # # Time array
    t = np.linspace(0, 10, 1000)

    # DH-Parameters
    n = 2  # DOF 
    # l = [l1, l2]
    l = np.array([1, 1])
    alpha = np.radians([0, 0])
    a = np.array([0, l[0]])
    d = np.array([0, 0])

    d_nn = Matrix([l[1], 0, 0])  # coord of EE wrt to last joint frame

    # Dyanamic Parameters
    m = np.array([1, 1])
    # m = [m1, m2]

    COG_wrt_body = []  # location of COG wrt to DH-frame / body frame
    MOI_about_body_CG = []  # MOI of the link about COG based on DH-frame / body frame
    for i in range(len(m)):
        COG_wrt_body.append(Matrix([[l[i]],[0],[0]]))
        # MOI_about_body_CG.append(Matrix([[0,          0         ,          0         ],
        #                                  [0, (m[i]/12)*(l[i]**2),          0         ],
        #                                  [0,          0         , (m[i]/12)*(l[i]**2)]]))
        MOI_about_body_CG.append(Matrix([[0,  0,  0],
                                         [0,  0,  0],
                                         [0,  0,  0]]))
        
    doosan = RNEA(n, COG_wrt_body, MOI_about_body_CG, d_nn, t)

    # Symbolic Matrices
    M_sym, C_sym, G_sym = doosan.mcg_matrix(alpha, a, d, m)

    # # Initial conditions
    q = np.array([[0],[0]])
    q_dot = np.array([[0],[0]])
    initial_state = [q[0,0], q_dot[0,0], q[1,0], q_dot[1,0]]
    
    # Solve ODE
    doosan.forward_dynamics(initial_state, t)
