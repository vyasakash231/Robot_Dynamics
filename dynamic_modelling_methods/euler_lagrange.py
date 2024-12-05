import os
import sys
from math import *
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *
from sympy.physics.vector import dynamicsymbols
import dill  # better than pickle for handling lambda functions
from collections import defaultdict

from .homo_transformation import *


class Euler_Lagrange:
    g = symbols("g")  # gravity
    def __init__(self, n, location_COM_wrt_body, MOI_wrt_body_CG, d_nn):
        self.n = n
        self.alpha = symarray('alpha',n)
        self.a = symarray('a',n)
        self.d = symarray('d',n)
        self.d_nn = d_nn

        self.t = Symbol("t")  # define time

        self.mass_vec = symarray('m',n)  # mass of each link
        self.G_vec = Matrix([[0],[Euler_Lagrange.g],[0]])  # Gravity Vector (gravity acting along -ve Y-axis)

        # Without time
        self.theta_vec = [symbols(f'theta_{i}') for i in range(n)]
        self.theta_dot_vec = [symbols(f'theta_dot_{i}') for i in range(n)]  # shape -> (n,)
        self.theta_ddot_vec = [symbols(f'theta_ddot_{i}') for i in range(n)]  # shape -> (n,)
        
        # With time
        self.theta_t_vec = [dynamicsymbols(f'theta_{i}') for i in range(n)]
        self.theta_t_dot_vec = [dynamicsymbols(f'theta_dot_{i}') for i in range(n)]  # shape -> (n,)
        
        self.b_ii = location_COM_wrt_body  # coordinates of COM of link from joint wrt body frame
        self.I_i_i = MOI_wrt_body_CG  # link MOI wrt body CG frame (as list of matrices)

        # Store symbols separately for each matrix
        self.M_symbols = self.theta_vec
        self.C_symbols = self.theta_vec + self.theta_dot_vec
        self.G_symbols = self.theta_vec

    def _data(self):
        # Import manipulator Transformation data using DH parameters (a_n_n -> coord of (i)th joint wrt (i-1)th joint)
        self.Rot_mat, self.O, self.a_i0 = symbolic_transformation_matrix(self.n, self.alpha, self.a, self.d, self.theta_t_vec, self.d_nn)
        
        # z-vector (last column) in rotation matrix 
        self.Z_n_0 = []  # (3,n)
        for R in self.Rot_mat:
            self.Z_n_0.append(R[:,[-1]])

        I_i_0 = []   # MOI of each link wrt base frame
        self.C_i0 = zeros(3,self.n)   # COM of each link wrt base frame
        self.C_dot_vec = zeros(3,self.n)
        self.omega_vec = zeros(3,self.n+1)  # Angular velocity of links wrt to ground frame
        
        subs_dict = defaultdict(list)
        for i in range(self.n):
            I_i_0.append(self.Rot_mat[i] * self.I_i_i[i] * self.Rot_mat[i].T)   # I = R * I * R.T
            self.C_i0[:,i] = self.O[:,i+1] + self.Rot_mat[i] * self.b_ii[i]   # coordinates of COM of link wrt base frame
            self.C_dot_vec[:,i] = diff(self.C_i0[:,i], self.t)   # time derivative
            self.omega_vec[:,i+1] = self.omega_vec[:,i] + self.Z_n_0[i] * self.theta_dot_vec[i]

            subs_dict[Derivative(self.theta_t_vec[i], self.t)] = self.theta_dot_vec[i]   # to replace Derivative(theta_{i}(t), t) with theta_dot_{i}
            subs_dict[self.theta_t_vec[i]] = self.theta_vec[i]   # to replace theta_{i}(t) with theta_{i}

        # convert to autonomous DS (time Independent) 
        self.I_i_0 = [i.subs(subs_dict) for i in I_i_0]
        self.C_i0 = self.C_i0.subs(subs_dict)
        self.C_dot_vec = self.C_dot_vec.subs(subs_dict)
        self.omega_vec = self.omega_vec.subs(subs_dict)
    
    def _kinetic_energy(self, alpha, a, d, mass):
        self.K_n = Matrix([[0]])
        for i in range(self.n):
            K_i = (1/2) * self.mass_vec[i] * transpose(self.C_dot_vec[:,i]) * self.C_dot_vec[:,i] + \
                  (1/2) * transpose(self.omega_vec[:,i+1]) * self.I_i_0[i] * self.omega_vec[:,i+1] 
            self.K_n += K_i

        self.K_n = self.K_n.subs([(k, alpha[idx]) for idx,k in enumerate(self.alpha)] + \
                                 [(k, a[idx]) for idx,k in enumerate(self.a)] + \
                                 [(k, d[idx]) for idx,k in enumerate(self.d)] + \
                                 [(k, mass[idx]) for idx,k in enumerate(self.mass_vec)])

    def _potential_energy(self, alpha, a, d, mass):
        self.P_n = Matrix([[0]])
        for i in range(self.n):
            P_i = -self.mass_vec[i] * transpose(self.C_i0[:,i]) * self.G_vec
            self.P_n += P_i

        self.P_n = self.P_n.subs([(k, alpha[idx]) for idx,k in enumerate(self.alpha)] + \
                                 [(k, a[idx]) for idx,k in enumerate(self.a)] + \
                                 [(k, d[idx]) for idx,k in enumerate(self.d)] + \
                                 [(k, mass[idx]) for idx,k in enumerate(self.mass_vec)])
        
    def lagrange(self, alpha, a, d, mass):
        self._data()
        self._kinetic_energy(alpha, a, d, mass)   # calculate Kinetic Energy 
        self._potential_energy(alpha, a, d, mass)   # calculate Potential Energy
        
        L = self.K_n - self.P_n

        dL_dtheta_dot = zeros(self.n,1)
        dL_dtheta = zeros(self.n,1)
        DdL_dtheta_dot_Dt = zeros(self.n,1)

        subs_dict_1 = defaultdict(list)
        for i in range(self.n):
            dL_dtheta_dot[i] = diff(L, self.theta_dot_vec[i])
            dL_dtheta[i] = diff(L, self.theta_vec[i])

            subs_dict_1[self.theta_dot_vec[i]] = self.theta_t_dot_vec[i]   # to replace theta_dot_{i} with theta_dot_{i}(t)
            subs_dict_1[self.theta_vec[i]] = self.theta_t_vec[i]   # to replace theta_{i} with theta_{i}(t) 

        # convert back to non-autonomous DS (time dependent)
        dL_dtheta_dot = dL_dtheta_dot.subs(subs_dict_1)
        dL_dtheta = dL_dtheta.subs(subs_dict_1)

        # -------- checked expressions till here -------- #
        
        subs_dict_2 = defaultdict(list)
        for i in range(self.n):
            DdL_dtheta_dot_Dt[i] = diff(dL_dtheta_dot[i], self.t)  # time derivative

            subs_dict_2[Derivative(self.theta_t_dot_vec[i], self.t)] = self.theta_ddot_vec[i]   # to replace Derivative(theta_dot_{i}(t), t) with theta_ddot_{i}
            subs_dict_2[Derivative(self.theta_t_vec[i], self.t)] = self.theta_dot_vec[i]   # to replace Derivative(theta_{i}(t), t) with theta_dot_{i}
            subs_dict_2[self.theta_t_dot_vec[i]] = self.theta_dot_vec[i]   # to replace theta_dot_{i}(t) with theta_dot_{i} 
            subs_dict_2[self.theta_t_vec[i]] = self.theta_vec[i]   # to replace theta_{i}(t) with theta_{i} 

        tau = DdL_dtheta_dot_Dt - dL_dtheta

        # -------- checked expressions till here -------- #

        tau = tau.subs(subs_dict_2)
        return tau

    def mcg_matrix(self, alpha, a, d, m):
        self.numeric_DH = [alpha, a, d]

        # substitute DH-parameter and mass in each equation
        tau_M = self.lagrange(alpha, a, d, m)  # calculate Lagrange 
        tau_C = self.lagrange(alpha, a, d, m)  # calculate Lagrange 
        tau_G = self.lagrange(alpha, a, d, m)  # calculate Lagrange 

        # C vector formulation
        self.C_vec = tau_C.subs([(k,0) for k in self.theta_ddot_vec] + [(Euler_Lagrange.g, 0)])  # substitute 0 for k --> (k,0)

        # M matrix formulation
        self.M = symarray('M',(self.n, self.n))
        for i in range(self.n):  # row
            for j in range(self.n):  # column
                if j == 0:
                    theta_ddot = Array(self.theta_ddot_vec[j+1:])
                elif j == self.n-1:
                    theta_ddot = Array(self.theta_ddot_vec[:j])
                else:
                    theta_ddot = Array(Array(self.theta_ddot_vec[:j]).tolist() + Array(self.theta_ddot_vec[j+1:]).tolist()) 
            
                M = tau_M[i].subs([(self.theta_ddot_vec[j], 1)] + [(k,0) for k in self.theta_dot_vec] + [(Euler_Lagrange.g, 0)])
                self.M[i,j] = M.subs([(k,0) for k in theta_ddot])  

        # G vector formulation
        self.G = tau_G.subs([(k,0) for k in self.theta_ddot_vec] + [(k,0) for k in self.theta_dot_vec] + [(Euler_Lagrange.g, -9.81)])

        C_mat = self.compute_coriolis_matrix()

        return simplify(self.M), simplify(self.C_vec), simplify(C_mat), simplify(self.G)  # symbolic
    
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
