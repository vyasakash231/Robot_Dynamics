from math import *
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import dill  # better than pickle for handling lambda functions
from collections import defaultdict

from .homo_transformation import *


class RNEA:
    g = symbols("g")  # gravity
    def __init__(self, n, location_COM_wrt_body, MOI_wrt_body_CG, d_nn, gravity_axis=[]):
        self.n = n
        self.alpha = symarray('alpha',n)
        self.a = symarray('a',n)
        self.d = symarray('d',n)
        self.d_nn = d_nn

        self.mass_vec = symarray('m',n)  # mass of each link
        self.G_vec = RNEA.g * gravity_axis  # Gravity Vector

        self.theta_vec = [symbols(f'theta_{i}') for i in range(n)]
        self.theta_dot_vec = [symbols(f'theta_dot_{i}') for i in range(n)]  # shape -> (n,)
        self.theta_ddot_vec = [symbols(f'theta_ddot_{i}') for i in range(n)]  # shape -> (n,)
        
        self.omega_vec = [zeros(3,1) for _ in range(self.n+1)]  # Angular velocity of links wrt to ground frame 
        self.omega_dot_vec = [zeros(3,1) for _ in range(self.n+1)]  # Angular acceleration of ground link wrt to ground frame 
        self.O_ddot_vec = [zeros(3,1) for _ in range(self.n+1)]  # Linear acceleration of the Joint/body frame wrt base frame

        self.C_ddot_vec = [zeros(3,1) for _ in range(self.n)]  # Linear acceleration of link COM

        # coordinates of COM of link wrt to body frame
        self.b_ii = location_COM_wrt_body 

        # Import manipulator Transformation data using DH parameters (a_n_n -> coord of (i)th joint wrt (i-1)th joint)
        self.Rot_mat, self.O, self.a_i0 = symbolic_transformation_matrix(self.n, self.alpha, self.a, self.d, self.theta_vec, self.d_nn)
        
        # z-vector (last column) in rotation matrix 
        self.Z_n_0 = []  # (3,n)
        for R in self.Rot_mat:
            self.Z_n_0.append(R[:,[-1]])
        
        self.I_i_i = MOI_wrt_body_CG  # link MOI wrt body CG frame (as list of matrices)
        self.I_i_0 = []  # MOI of each link wrt base frame
        for i in range(self.n):
            self.I_i_0.append(self.Rot_mat[i] * self.I_i_i[i] * self.Rot_mat[i].T)  # I = R * I * R.T

        # Store symbols separately for each matrix
        self.M_symbols = self.theta_vec
        self.C_symbols = self.theta_vec + self.theta_dot_vec
        self.G_symbols = self.theta_vec

    def _forward_pass(self):
        self.b_i0 = zeros(3,self.n+1)
        self.f_i_0 = [zeros(3,1) for _ in range(self.n)]  # Net-force of (i)th link COM wrt to base frame
        self.n_i_0 = [zeros(3,1) for _ in range(self.n)]  # Net-moment of (i)th link wrt to base frame
        self.r_i0 = zeros(3,self.n)
        
        for i in range(self.n):
            self.omega_vec[i+1] = self.omega_vec[i] + self.Z_n_0[i] * self.theta_dot_vec[i]  # Angular velocity of links wrt to ground frame
            
            self.b_i0[:,i+1] = self.Rot_mat[i] * self.b_ii[i]
            
            ext_1 = self.Z_n_0[i] * self.theta_dot_vec[i]
            self.omega_dot_vec[i+1] = self.omega_dot_vec[i] + self.omega_vec[i+1].cross(ext_1) + self.Z_n_0[i] * self.theta_ddot_vec[i]  # Angular acceleration
            
            ext_2 = self.O[:,i+1] - self.O[:,i]
            ext_3 = self.omega_vec[i].cross(ext_2)
            self.O_ddot_vec[i+1] = self.O_ddot_vec[i] + self.omega_dot_vec[i].cross(ext_2) + self.omega_vec[i].cross(ext_3)  # Linear acceleration of joints
            
            ext_4 = self.omega_vec[i+1].cross(self.b_i0[:,i+1])
            self.C_ddot_vec[i] = self.O_ddot_vec[i+1] + self.omega_dot_vec[i+1].cross(self.b_i0[:,i+1]) + self.omega_vec[i+1].cross(ext_4) # Linear acceleration of COM
            
            '''D-Alembert demonstrated that by adding “inertial force” and “inertial torque” or moment, 
            an accelerating rigid body can be transformed into an equal static system. 
            The inertial torque can act anywhere and the inertial force must act through the centre of mass.'''
            self.f_i_0[i] = self.mass_vec[i] * self.C_ddot_vec[i]  # Inertial force acting on COM (if we consider D’Alembert’s Principle)
            
            self.n_i_0[i] = self.I_i_0[i] * self.omega_dot_vec[i+1] + self.omega_vec[i+1].cross(self.I_i_0[i] * self.omega_vec[i+1])  # Inertial moment / torque on each joint
            
            self.r_i0[:,i] = self.a_i0[:,i+1] - self.b_i0[:,i+1]  # r_i0 = r_{i}_0

    def _backward_pass(self):
        self.f_j_0 = [zeros(3,1) for _ in range(self.n+1)]  # Net-force applied by (n-1)th link on last / (n)th link 
        self.n_j_0 = [zeros(3,1) for _ in range(self.n+1)]  # Net-moment applied by (n-1)th link on last / (n)th link 
        self.tau = zeros(self.n,1)   # Driving Torque
        
        for j in range(self.n-1, -1, -1):  # [n, n-1, n-2, n-3, ......, 0]
            self.f_j_0[j] = self.f_i_0[j] + self.f_j_0[j+1] - (self.mass_vec[j] * self.G_vec)
            self.n_j_0[j] = self.n_i_0[j] + self.n_j_0[j+1] + self.b_i0[:,j+1].cross(self.f_j_0[j]) + self.r_i0[:,j].cross(self.f_j_0[j+1])
            self.tau[j] = self.Z_n_0[j].T * self.n_j_0[j]  # Driving Torque

    """Joint Space"""
    def mcg_jointspace(self, alpha, a, d, m):
        self.numeric_DH = [alpha, a, d]

        self._forward_pass()  # Perform forward pass
        self._backward_pass()  # Perform backward pass

        # substitute DH-parameter and mass in each equation
        tau = self.tau.subs([(k, alpha[idx]) for idx,k in enumerate(self.alpha)] + \
                            [(k, a[idx]) for idx,k in enumerate(self.a)] + \
                            [(k, d[idx]) for idx,k in enumerate(self.d)] + \
                            [(k, m[idx]) for idx,k in enumerate(self.mass_vec)])

        tau_C = tau.copy()
        tau_M = tau.copy()
        tau_G = tau.copy()

        # C matrix formulation
        self.C_vec = tau_C.subs([(k,0) for k in self.theta_ddot_vec] + [(RNEA.g, 0)])  # substitute 0 for k --> (k,0)
        
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

        C_mat = self.compute_coriolis_matrix()

        return simplify(self.M), simplify(self.C_vec), simplify(C_mat), simplify(self.G)  # symbolic
    
    """Joint Space"""
    def compute_coriolis_matrix(self):
        # C matrix formulation
        C = symarray('M',(self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                a_terms = 0
                b_terms = 0
                for k in range(self.n):
                    a_terms += diff(self.M[i,j], self.theta_vec[k]) * self.theta_dot_vec[k]
                    b_terms += (diff(self.M[i, k], self.theta_vec[j]) - diff(self.M[j, k], self.theta_vec[i])) * self.theta_dot_vec[k]
                C[i, j] = (1/2) * (a_terms + b_terms)
        return C
    
    def mcg_taskspace(self):
        self.Jacobian = symbolic_Jacobian(self.n, self.alpha, self.a, self.d, self.theta_vec, self.d_nn)
    
    def save_equations(self, M_sym, C_vec_sym, C_mat_sym, G_sym, filename='lagrangian_eqs'):
        """Save symbolic equations to a file"""
        # Converts the symbolic expressions into lambda functions 
        M_lambda = lambdify(self.M_symbols, M_sym, 'numpy')
        C_vec_lambda = lambdify(self.C_symbols, C_vec_sym, 'numpy')
        C_mat_lambda = lambdify(self.C_symbols, C_mat_sym, 'numpy')
        G_lambda = lambdify(self.G_symbols, G_sym, 'numpy')
        
        # Store these lambda functions in a dictionary along with the original symbolic expressions and the corresponding symbol lists
        equations = {
            'M_lambda': M_lambda,
            'C_vec_lambda': C_vec_lambda,
            'C_mat_lambda': C_mat_lambda,
            'G_lambda': G_lambda,
            'M_sym': M_sym,
            'C_vec_sym': C_vec_sym,
            'C_mat_sym': C_mat_sym,
            'G_sym': G_sym,
            'M_symbols': self.M_symbols,
            'C_symbols': self.C_symbols,
            'G_symbols': self.G_symbols
        }
        
        with open('../models/'+filename+'.pkl', 'wb') as f:
            dill.dump(equations, f)
    
    @staticmethod
    def load_equations(filename='lagrangian_eqs'):
        """Load equations from file"""
        with open('../models/'+filename+'.pkl', 'rb') as f:
            return dill.load(f)