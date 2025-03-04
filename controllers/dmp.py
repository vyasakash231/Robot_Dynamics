import sys
import os
import numpy as np
import scipy.interpolate
import scipy.linalg
import copy
np.set_printoptions(suppress=True)

from .utils import *

# DMP Explained : https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/
class DMP:
    def __init__(self, no_of_DMPs, no_of_basis_func, dt=0.01, T=1, X_0=None, X_g=None, alpha=3, K=1050, D=None, W=None, method=None, obstacles=None):
        """
        no_of_DMPs         : number of dynamic movement primitives (i.e. dimensions)
        no_of_basis_func   : number of basis functions per DMP (actually, they will be one more)
        dt             : timestep for simulation
        X_0            : initial state of DMPs
        X_g            : X_g state of DMPs
        T              : final time
        K              : elastic parameter in the dynamical system
        D              : damping parameter in the dynamical system
        w              : associated weights
        alpha          : constant of the Canonical System
        """
        self.no_of_DMPs = no_of_DMPs
        self.no_of_basis_func = no_of_basis_func

        # Set up the DMP system
        if X_0 is None:
            X_0 = np.zeros(self.no_of_DMPs)
        self.X_0 = copy.deepcopy(X_0)

        if X_g is None:
            X_g = np.ones(self.no_of_DMPs)
        self.X_g = copy.deepcopy(X_g)

        self.K = K  # stiffness
        if D is None:
            self.D = 2 * np.sqrt(self.K)  # damping 
        else:
            self.D = D

        self.cs = Canonical_System(dt=dt, alpha=alpha, run_time=T)  # setup a canonical system
        
        self.reset_state()  # set up the DMP system

        self.center_of_gaussian()  # centers of Gaussian basis functions distributed along the phase of the movement
        self.variance_of_gaussian()  # width/variance of Gaussian basis functions distributed along the phase of the movement
        
        # If no weights are give, set them to zero (default, f = 0)
        if W is None:  
            W = np.zeros((self.no_of_DMPs, self.no_of_basis_func))
        self.W = W

        """Define Obstacle"""
        if obstacles is not None:
            if method == "DS":
                self.modulation = Modulation_2012()
                # self.modulation = Modulation_2019()
                self.obstacles = obstacles
            if method == "APF":
                self.obstacles = Obstacle_Static(obstacles, A=5.0, eta=1.0)
        else:
            self.obstacles = None

    def center_of_gaussian(self):
        self.c = np.exp(-self.cs.alpha * np.linspace(0, self.cs.run_time, self.no_of_basis_func + 1))  #  centers are exponentially spaced

    def variance_of_gaussian(self):
        """width/variance of gaussian distribution"""
        self.width = np.zeros(self.no_of_basis_func)
        for i in range(self.no_of_basis_func):
            self.width[i] = 1 / ((self.c[i+1] - self.c[i])**2)
        self.width = np.append(self.width, self.width[-1])                                                                                              

    def reset_state(self):
        """Reset the system state"""
        self.X = self.X_0.copy()
        self.dX = np.zeros((self.no_of_DMPs, 1))
        self.ddX = np.zeros((self.no_of_DMPs, 1))
        self.cs.reset()

    def gaussian_basis_func(self, theta):
        """Generates the activity of the basis functions for a given canonical system rollout"""
        c = np.reshape(self.c, [self.no_of_basis_func + 1, 1])
        h = np.reshape(self.width, [self.no_of_basis_func + 1, 1])
        Psi_basis_func = np.exp(-h * (theta - c)**2)
        return Psi_basis_func

    def generate_weights(self, f_target, theta_track):
        """
        Generate a set of weights over the basis functions such that the target forcing 
        term trajectory is matched (f_target - f(θ), shape -> [no_of_DMPs x time_steps])

                   | /  ψ(θ)  \     |^(-1)
        W = f(θ) * ||----------| * θ|
                   | \ ∑ ψ(θ) /     |
        """
        # generate Basis functions
        psi = self.gaussian_basis_func(theta_track)

        # calculate basis function weights using "linear regression"
        sum_psi = np.sum(psi,0)
        self.W = np.nan_to_num(f_target.T @ np.linalg.pinv((psi / sum_psi) * theta_track))

    def learn_dynamics(self, X_des):
        """
        Takes in a desired trajectory and generates the set of system parameters that best realize this path.
        X_des: the desired trajectories of each DMP should be shaped [no_of_dmps, num_timesteps]
        """
        # set initial and goal state
        self.X_0 = X_des[:,[0]].copy()  # [[x0],[y0]]
        self.X_g = X_des[:,[-1]].copy()  # [[xn],[yn]]

        t_des = np.linspace(0, self.cs.run_time, X_des.shape[1])  # demo trajectory timing
        std_time = np.linspace(0, self.cs.run_time, self.cs.time_steps)  # map demo timing to standard time

        # --------------------------------  Using Vector ---------------------------------------
        path_gen = scipy.interpolate.interp1d(t_des, X_des, kind="quadratic")

        # Evaluation of the interpolant
        path = path_gen(std_time)
        # --------------------------------------------------------------------------------------
        
        X_des = path  # [[x0,x1,x2....], [y0,y1,y2,....]] -> (2,101)
        
        # calculate velocity of y_des (gradient of X_des is computed using second order accurate central differences)
        dX_des = np.gradient(X_des, self.cs.dt, axis=1, edge_order=2)  # (2,101)
        
        """https://stackoverflow.com/questions/24633618/what-does-numpy-gradient-do"""

        # calculate acceleration of y_des (gradient of dX_des is computed using second order accurate central differences)
        ddX_des = np.gradient(dX_des, self.cs.dt, axis=1, edge_order=2)  # (2,101)
       
        theta_track = self.cs.rollout()
    
        ## Find the force required to move along this trajectory
        """
        this is equation (11) from paper,
        D. -H. Park, H. Hoffmann, P. Pastor and S. Schaal, "Movement reproduction and obstacle avoidance with dynamic movement primitives and potential fields," 
        Humanoids 2008 - 8th IEEE-RAS International Conference on Humanoid Robots, Daejeon, Korea (South), 2008, pp. 91-98, doi: 10.1109/ICHR.2008.4755937.        
        """
        f_target = np.zeros([self.cs.time_steps, self.no_of_DMPs])
        for idx in range(self.no_of_DMPs):
            f_target[:,idx] = (ddX_des[idx,:] / self.K) - (self.X_g[idx] - X_des[idx,:]) + (self.D / self.K) * dX_des[idx,:] + (self.X_g[idx] - self.X_0[idx]) * theta_track  # (101,2)
        
        # generate weights to realize f_target
        self.generate_weights(f_target, theta_track)
        self.reset_state()
    
    def step(self, X_g, gamma, tau=1):
        """
        DMP 2nd order system in vector form (for 3 DOF system);
        τ*dV = K*(X_g - X) - D*V - K*(X_g - X_0)*θ + K*f
        τ*dX = V

        In matrix form;
            |dV_x|   |K 0 0|   |g_x - X_x|   |D 0 0|   |V_x|   |K 0 0|   |g_x - X_x0|       |K 0 0|   |f_x|
        τ * |dV_y| = |0 K 0| * |g_y - X_y| - |0 D 0| * |V_y| - |0 K 0| * |g_y - X_y0| * θ + |0 K 0| * |f_y|       
            |dV_z|   |0 0 K|   |g_z - X_z|   |0 0 D|   |V_z|   |0 0 K|   |g_z - X_z0|       |0 0 K|   |f_z|      

            |dX_x|   |V_x|
        τ * |dX_y| = |V_y|
            |dX_z|   |V_z|

        State-Space form for 3 DOF/No_of_DMPs system;
        state_vector, Y = [y1, y2, y3, y4, y5, y6] = [V_x, X_x, V_y, X_y, V_z, X_z]
        
                |dy1|   |dV_x|         |-D -K  0  0  0  0|   |V_x|         |g_x - (g_x - X_x0)*θ + f_x|
                |dy2|   |dX_x|         | 1  0  0  0  0  0|   |X_x|         |           0              |
        dY_dt = |dy3| = |dV_y| = 1\τ * | 0  0 -D -K  0  0| * |V_y| + K\τ * |g_y - (g_y - X_y0)*θ + f_y| 
                |dy4|   |dX_y|         | 0  0  1  0  0  0|   |X_y|         |           0              |
                |dy5|   |dV_z|         | 0  0  0  0 -D -K|   |V_z|         |g_z - (g_z - X_z0)*θ + f_z|
                |dy6|   |dX_z|         | 0  0  0  0  1  0|   |X_z|         |           0              |

        dY_dt = A @ Y + B   (A-matrix must have constant coeff for DS to be linear)
        """

        # define state vector (Y)
        Y = np.zeros((2*self.no_of_DMPs, 1))  # Y = [[0], [0], [0], [0]]
        Y[range(0,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.dX)  # [[Vx], [0], [Vy], [0]]
        Y[range(1,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.X)  # [[Vx], [x], [Vy], [y]]
        
        # define A-matrix
        A = np.zeros((2*self.no_of_DMPs, 2*self.no_of_DMPs))
        A[range(0, 2*self.no_of_DMPs, 2), range(0, 2*self.no_of_DMPs, 2)] = -self.D / tau
        A[range(0, 2*self.no_of_DMPs, 2), range(1, 2*self.no_of_DMPs, 2)] = -self.K / tau
        A[range(1, 2*self.no_of_DMPs, 2), range(0, 2*self.no_of_DMPs, 2)] = 1 / tau

        """Run the DMP system for a single timestep"""
        psi = self.gaussian_basis_func(self.cs.theta)  # update basis function
        
        # update forcing term using weights learnt while imitating given trajectory
        sum_psi = np.sum(psi[:,[0]])
        if np.abs(sum_psi) <= 1e-6:  # avoid division by 0
            f = 0.0 * np.dot(self.W, psi[:,[0]])
        else:
            f = (np.dot(self.W, psi[:,[0]]) / sum_psi) * self.cs.theta

        # define B-matrix
        B = np.zeros((2 * self.no_of_DMPs ,1))
        B[0::2,:] = (self.K/tau) * (X_g - (X_g - self.X_0) * self.cs.theta + f)

        # solve above dynamical system using Euler-forward method / Runge-kutta 4th order / Exponential Integrators 
        dY_dt = rk4_step(Y,A,B,gamma*self.cs.dt)
        # dY_dt = forward_euler(Y,A,B,self.cs.dt)

        Y = Y + dY_dt * (gamma*self.cs.dt)
        
        # extract position-X, velocity-V, acceleration data from current state vector-Y values
        self.X = Y[1::2, :]   # extract position data from state vector Y
        self.dX = Y[0::2, :] # extract velocity data from state vector Y
        
        self.cs.step(tau=tau)  # update theta
        return self.X, self.dX   
    
    def step_with_APF(self, X_g, gamma, tau=1):
        """
        DMP 2nd order system in vector form (for 3 DOF system);
        τ*dV = K*(X_g - X) - D*V - K*(X_g - X_0)*θ + K*f
        τ*dX = V

        In matrix form;
            |dV_x|   |K 0 0|   |g_x - X_x|   |D 0 0|   |V_x|   |K 0 0|   |g_x - X_x0|       |K 0 0|   |f_x|
        τ * |dV_y| = |0 K 0| * |g_y - X_y| - |0 D 0| * |V_y| - |0 K 0| * |g_y - X_y0| * θ + |0 K 0| * |f_y|       
            |dV_z|   |0 0 K|   |g_z - X_z|   |0 0 D|   |V_z|   |0 0 K|   |g_z - X_z0|       |0 0 K|   |f_z|      

            |dX_x|   |V_x|
        τ * |dX_y| = |V_y|
            |dX_z|   |V_z|

        State-Space form for 3 DOF/No_of_DMPs system;
        state_vector, Y = [y1, y2, y3, y4, y5, y6] = [V_x, X_x, V_y, X_y, V_z, X_z]
        
                |dy1|   |dV_x|         |-D -K  0  0  0  0|   |V_x|         |g_x - (g_x - X_x0)*θ + f_x|
                |dy2|   |dX_x|         | 1  0  0  0  0  0|   |X_x|         |           0              |
        dY_dt = |dy3| = |dV_y| = 1\τ * | 0  0 -D -K  0  0| * |V_y| + K\τ * |g_y - (g_y - X_y0)*θ + f_y| 
                |dy4|   |dX_y|         | 0  0  1  0  0  0|   |X_y|         |           0              |
                |dy5|   |dV_z|         | 0  0  0  0 -D -K|   |V_z|         |g_z - (g_z - X_z0)*θ + f_z|
                |dy6|   |dX_z|         | 0  0  0  0  1  0|   |X_z|         |           0              |

        dY_dt = A @ Y + B   (A-matrix must have constant coeff for DS to be linear)
        """

        # define state vector (Y)
        Y = np.zeros((2*self.no_of_DMPs, 1))  # Y = [[0], [0], [0], [0]]
        Y[range(0,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.dX)  # [[Vx], [0], [Vy], [0]]
        Y[range(1,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.X)  # [[Vx], [x], [Vy], [y]]
        
        # define A-matrix
        A = np.zeros((2*self.no_of_DMPs, 2*self.no_of_DMPs))
        A[range(0, 2*self.no_of_DMPs, 2), range(0, 2*self.no_of_DMPs, 2)] = -self.D / tau
        A[range(0, 2*self.no_of_DMPs, 2), range(1, 2*self.no_of_DMPs, 2)] = -self.K / tau
        A[range(1, 2*self.no_of_DMPs, 2), range(0, 2*self.no_of_DMPs, 2)] = 1 / tau

        """Run the DMP system for a single timestep"""
        psi = self.gaussian_basis_func(self.cs.theta)  # update basis function
        
        # update forcing term using weights learnt while imitating given trajectory
        sum_psi = np.sum(psi[:,[0]])
        if np.abs(sum_psi) <= 1e-6:  # avoid division by 0
            f = 0.0 * np.dot(self.W, psi[:,[0]])
        else:
            f = (np.dot(self.W, psi[:,[0]]) / sum_psi) * self.cs.theta

        # define B-matrix
        B = np.zeros((2 * self.no_of_DMPs ,1))
        B[0::2,:] = (self.K/tau) * (X_g - (X_g - self.X_0) * self.cs.theta + f)

        if self.obstacles is not None:
            B[0::2,:] += self.obstacles.gen_external_force(self.X) / tau 

        # solve above dynamical system using Euler-forward method / Runge-kutta 4th order / Exponential Integrators 
        dY_dt = rk4_step(Y,A,B,gamma*self.cs.dt)
        # dY_dt = forward_euler(Y,A,B,self.cs.dt)

        Y = Y + dY_dt * (gamma*self.cs.dt)
        
        # extract position-X, velocity-V, acceleration data from current state vector-Y values
        self.X = Y[1::2, :]   # extract position data from state vector Y
        self.dX = Y[0::2, :] # extract velocity data from state vector Y
        
        self.cs.step(tau=tau)  # update theta
        return self.X, self.dX   

    def step_with_DS_2012(self, X_g, gamma, tau=1):
        """
        DMP 2nd order system in vector form (for 3 DOF system);
        τ*dV = K*(X_g - X) - D*[M*V] - K*(X_g - X_0)*θ + K*f
        τ*dX = M*V

        In matrix form;
            |dV_x|   |K 0 0|   |g_x - X_x|   |D 0 0|    /|M_11 M_12 M_13|   |V_x|\    |K 0 0|   |g_x - X_x0|       |K 0 0|   |f_x|
        τ * |dV_y| = |0 K 0| * |g_y - X_y| - |0 D 0| * | |M_21 M_22 M_23| * |V_y| | - |0 K 0| * |g_y - X_y0| * θ + |0 K 0| * |f_y|       
            |dV_z|   |0 0 K|   |g_z - X_z|   |0 0 D|    \|M_31 M_32 M_33|   |V_z|/    |0 0 K|   |g_z - X_z0|       |0 0 K|   |f_z|      

            |dX_x|   |M_11 M_12 M_13|   |V_x|
        τ * |dX_y| = |M_21 M_22 M_23| * |V_y|
            |dX_z|   |M_31 M_32 M_33|   |V_z|

        State-Space form for 3 DOF/No_of_DMPs system;
        state_vector, Y = [y1, y2, y3, y4, y5, y6] = [V_x, X_x, V_y, X_y, V_z, X_z]
        
                |dy1|   |dV_x|         |-D*M_11 -K -D*M_12  0  -D*M_13  0|   |V_x|         |g_x - (g_x - X_x0)*θ + f_x|
                |dy2|   |dX_x|         |  M_11   0   M_12   0    M_13   0|   |X_x|         |           0              |
        dY_dt = |dy3| = |dV_y| = 1\τ * |-D*M_21  0 -D*M_22 -K  -D*M_23  0| * |V_y| + K\τ * |g_y - (g_y - X_y0)*θ + f_y| 
                |dy4|   |dX_y|         |  M_21   0   M_22   0    M_23   0|   |X_y|         |           0              |
                |dy5|   |dV_z|         |-D*M_31  0 -D*M32   0  -D*M_33 -K|   |V_z|         |g_z - (g_z - X_z0)*θ + f_z|
                |dy6|   |dX_z|         |  M_31   0   M_32   0    M_33   0|   |X_z|         |           0              |

        dY_dt = A @ Y + B   (A-matrix must have constant coeff for DS to be linear)
        """

        # define state vector (Y)
        Y = np.zeros((2*self.no_of_DMPs, 1))  # Y = [[0], [0], [0], [0], [0], [0]]
        Y[range(0,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.dX)  # [[Vx], [0], [Vy], [0], [Vz], [0]]
        Y[range(1,2*self.no_of_DMPs, 2),:] = copy.deepcopy(self.X)  # [[Vx], [x], [Vy], [y], [Vz], [z]]

        # find modulation matrix
        M = self.modulation.compute_modulation_matrix(self.X, self.dX, self.obstacles)
        
        # define A-matrix
        A = np.zeros((2*self.no_of_DMPs, 2*self.no_of_DMPs))

        # Using index lists
        rows_1 = cols_1 = [0, 2, 4]
        A[np.ix_(rows_1, cols_1)] = (-self.D / tau) * M

        rows_2 = [1, 3, 5]
        cols_2 = [0, 2, 4]
        A[np.ix_(rows_2, cols_2)] = (1 / tau) * M

        A[range(0, 2*self.no_of_DMPs, 2), range(1, 2*self.no_of_DMPs, 2)] = -self.K / tau

        """Run the DMP system for a single timestep"""
        psi = self.gaussian_basis_func(self.cs.theta)  # update basis function
        
        # update forcing term using weights learnt while imitating given trajectory
        sum_psi = np.sum(psi[:,[0]])
        if np.abs(sum_psi) <= 1e-6:  # avoid division by 0
            f = 0.0 * np.dot(self.W, psi[:,[0]])
        else:
            f = (np.dot(self.W, psi[:,[0]]) / sum_psi) * self.cs.theta

        # define B-matrix
        B = np.zeros((2 * self.no_of_DMPs ,1))
        B[0::2,:] = (self.K/tau) * (X_g - (X_g - self.X_0) * self.cs.theta + f)

        # solve above dynamical system using Euler-forward method / Runge-kutta 4th order / Exponential Integrators 
        dY_dt = rk4_step(Y,A,B,gamma*self.cs.dt)
        # dY_dt = forward_euler(Y,A,B,self.cs.dt)

        Y = Y + dY_dt * (gamma*self.cs.dt)
        
        # extract position-X, velocity-V, acceleration data from current state vector-Y values
        self.X = Y[1::2, :]   # extract position data from state vector Y
        self.dX = Y[0::2, :] # extract velocity data from state vector Y
        
        self.cs.step(tau=tau)  # update theta
        return self.X, self.dX 

