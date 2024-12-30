import numpy as np
import cvxpy as cp
from colorama import Fore, Back, Style  # available color (RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)

class Linear_MPC_JointSpace:
    """
    Robot Linear Model: Double Integrator 
        e_ddot = (q̈_des - q̈) = u
    let's convert this into state-space form, 
    x1 = (q_des - q) = e 
    x2 = (q̇_des - q̇) = e_dot
    
    taking derivative,
    ẋ1 = e_dot = x2 -- (1)
    ẋ2 = e_ddot = u -- (2)

    so, final state-space form,
    |ẋ1|   |0 I| |x1|   |0|               |0 I|
    |  | = |   |*|  | + | |*u, where Ac = |   |
    |ẋ2|   |0 0| |x2|   |I|               |0 0|

    to, discretizing the continuous-time system, we'll integrate eqn (1) and (2),
    ∫ẋ2 = ∫u  --> ∫dx2 = ∫u*dτ = u∫dτ = u*[(t+dt) - t] = u*dt
                so, x2(t+dt) - x2(t) = u*dt
                    x2(t+dt) = x2(t) + u*dt

    ∫ẋ1 = ∫x2  --> ∫dx1 = ∫x2(τ)*dτ = ∫[x₂(t) + (τ-t)u(t)]*dτ = x2(t)*dt + (dt^2)/2 * u
                so, x1(t+dt) - x1(t) = x2(t)*dt + (dt²)/2 * u
                    x1(t+dt) = x1(t) + x2(t)*dt + (dt²)/2 * u

    therefore, the exact discrete-time system is,
    |x1(t+h)|   |I h*I| |x1(t)|   |(h²/2)*I |                  |I  h*I| 
    |       | = |     |*|     | + |         |*u(t), where Ad = |      |
    |x2(t+h)|   |0   I| |x2(t)|   |    dt   |                  |0   I |
    """
    def __init__(self, n, x_ref, N=20, state_cost=5.0, input_cost=1.0):
        self.n = n  # number of joints
        self.h = 0.1  
        self.N = N   # prediction horizon
        
        # State and input dimensions
        self.n_state = 2 * n  # [q; q_dot]
        self.n_input = n      # joint accelerations

        # desired trajectory
        self.x_ref = x_ref
        self.q_ref = x_ref[:n,:]  # desired joint position
        self.q_dot_ref = x_ref[n:,:]  # desired joint velocity

        self.step = 0
        
        # Discretize the continuous-time system using Zero-Order Hold (ZOH)
        self.Ad = np.block([[np.eye(n, n), self.h * np.eye(n, n)],
                            [np.zeros((n, n)), np.eye(n, n)]])  # (2nx2n)
        
        self.Bd = np.block([[((self.h**2)/2) * np.eye(n, n)], [self.h * np.eye(n, n)]])  # (2nxn)
        
        # MPC weights
        Q = np.ones((self.n_state)) * state_cost    # State cost
        R = np.ones((self.n_input)) * input_cost   # Input cost

        self.Q = np.diag(np.tile(Q, self.N))
        self.R = np.diag(np.tile(R, self.N))
        
    def set_joint_limits(self, q_min, q_max, q_dot):
        """Set joint position limits"""
        self.q_min = q_min
        self.q_max = q_max
        self.q_dot_min = -q_dot
        self.q_dot_max = q_dot

        self._precompute_mpc_matrices()  # Precompute matrices for QP formulation

    def _precompute_mpc_matrices(self):
        # Prediction matrices
        self.A = np.zeros((self.N * self.n_state, self.n_state))
        self.B = np.zeros((self.N * self.n_state, self.N * self.n_input))
        
        temp_A = self.Ad
        for i in range(self.N):
            self.A[i * self.n_state:(i + 1) * self.n_state, :] = np.linalg.matrix_power(temp_A, i+1)

        B_stack = self.Bd
        for j in range(1,self.N):  # rows
            for k in range(0,j+1):  # columns
                A_pow = np.linalg.matrix_power(self.Ad, j-k)
                B_stack = np.hstack((B_stack, A_pow @ self.Bd))

        p1, p2 = 1, 0
        for l in range(0, self.n_state*self.N, self.n_state):  # range(start, stop, step) 
            for m in range(0, p1, self.n_input):
                self.B[l:l+self.n_state, m:m+self.n_input] = B_stack[:,p2:p2+self.n_input]
                p2 += self.n_input
            p1 += self.n_input
        
    def compute_control(self, x_current):
        """
        Compute optimal control using quadratic programming
        x0: current state error [qd - q; qd_dot - q_dot]
        """
        # Define optimization variables
        U = cp.Variable((self.n_input*(self.N), 1))

        # Define Cost Function
        E0 = self.x_ref[:, [self.step]] - x_current
        E = self.A @ E0 + self.B @ U

        E_cost = cp.quad_form(E, self.Q)

        U_cost = cp.quad_form(U, self.R)

        cost = E_cost + U_cost

        constraints = []

        # State Constraints
        for i in range(self.N):  
            for j in range(self.n):
                q_ref = self.q_ref[j, i + (self.step + 1)]
                # Position Lower Bound
                constraints.append(q_ref - (self.A[i*self.n_state + j, :] @ E0 + self.B[i*self.n_state + j, :] @ U) >= self.q_min[j])

                # Position upper Bound
                constraints.append(q_ref - (self.A[i*self.n_state + j, :] @ E0 + self.B[i*self.n_state + j, :] @ U) <= self.q_max[j])

        prob = cp.Problem(cp.Minimize(cost), constraints)

        try:
            # Set verbose to True to get more information about solver
            prob.solve(verbose=False)
        except Exception as e:
            print(f"Optimization error: {e}")
            raise

        self.step += 1

        if prob.status == cp.OPTIMAL:
            # Extract optimal control input 
            u_optimal = U.value[:self.n_input]
            print(Fore.GREEN + f"Optimization Status: {prob.status}")
            print(Style.RESET_ALL)
            return u_optimal
        
        elif prob.status == cp.INFEASIBLE:
            print(Fore.RED + "Optimization problem is INFEASIBLE")
            print(Style.RESET_ALL)
            return np.zeros((self.n_input,1))
        
        else:
            print(f"Unexpected problem status: {prob.status}")
            raise ValueError(f"Optimization problem: {prob.status}")


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


class Linear_MPC_TaskSpace:
    """
    Robot Linear Model: Double Integrator 
            Mx * a = F  
        a = Mx^-1 * F = u
            a = u
    let's convert this into state-space form, 
            x1 = X
            x2 = ω
    
    taking derivative,
            ẋ1 = x2 -- (1)
            ẋ2 = u -- (2)

    so, final state-space form,
    |ẋ1|   |0 I| |x1|   |0|               |0 I|
    |  | = |   |*|  | + | |*u, where Ac = |   |
    |ẋ2|   |0 0| |x2|   |I|               |0 0|

    to, discretizing the continuous-time system, we'll integrate eqn (1) and (2),
    ∫ẋ2 = ∫u  --> ∫dx2 = ∫u*dτ = u∫dτ = u*[(t+dt) - t] = u*dt
                so, x2(t+dt) - x2(t) = u*dt
                    x2(t+dt) = x2(t) + u*dt

    ∫ẋ1 = ∫x2  --> ∫dx1 = ∫x2(τ)*dτ = ∫[x₂(t) + (τ-t)u(t)]*dτ = x2(t)*dt + (dt^2)/2 * u
                so, x1(t+dt) - x1(t) = x2(t)*dt + (dt²)/2 * u
                    x1(t+dt) = x1(t) + x2(t)*dt + (dt²)/2 * u

    therefore, the exact discrete-time system is,
    |x1(t+h)|   |I h*I| |x1(t)|   |(h²/2)*I |                  |I  h*I| 
    |       | = |     |*|     | + |         |*u(t), where Ad = |      |
    |x2(t+h)|   |0   I| |x2(t)|   |    dt   |                  |0   I |
    """
    def __init__(self, n, x_ref, N=20, state_cost=5.0, input_cost=1.0):
        self.n = 3  # no of states in taskspace
        self.h = 0.1
        self.N = N   # prediction horizon
        
        # State and input dimensions
        self.n_state = 2*self.n  # [x; x_dot]
        self.n_input = self.n      # end-effector forces

        # desired trajectory
        self.x_ref = x_ref
        self.q_ref = x_ref[:n,:]  # desired joint position
        self.q_dot_ref = x_ref[n:,:]  # desired joint velocity

        self.step = 0
        
        # Discretize the continuous-time system using Zero-Order Hold (ZOH)
        self.Ad = np.block([[np.eye(self.n, self.n), self.h * np.eye(self.n, self.n)],
                            [np.zeros((self.n, self.n)), np.eye(self.n, self.n)]])  # (2nx2n)
        
        self.Bd = np.block([[((self.h**2)/2) * np.eye(self.n, self.n)], [self.h * np.eye(self.n, self.n)]])  # (2nxn)
        
        # MPC weights
        Q = np.ones((self.n_state)) * state_cost    # State cost
        R = np.ones((self.n_input)) * input_cost   # Input cost

        self.Q = np.diag(np.tile(Q, self.N))
        self.R = np.diag(np.tile(R, self.N))
        
    def set_joint_limits(self, q_min, q_max, q_dot):
        self._precompute_mpc_matrices()  # Precompute matrices for QP formulation

    def _precompute_mpc_matrices(self):
        # Prediction matrices
        self.A = np.zeros((self.N * self.n_state, self.n_state))
        self.B = np.zeros((self.N * self.n_state, self.N * self.n_input))
        
        temp_A = self.Ad
        for i in range(self.N):
            self.A[i * self.n_state:(i + 1) * self.n_state, :] = np.linalg.matrix_power(temp_A, i+1)

        B_stack = self.Bd
        for j in range(1,self.N):  # rows
            for k in range(0,j+1):  # columns
                A_pow = np.linalg.matrix_power(self.Ad, j-k)
                B_stack = np.hstack((B_stack, A_pow @ self.Bd))

        p1, p2 = 1, 0
        for l in range(0, self.n_state*self.N, self.n_state):  # range(start, stop, step) 
            for m in range(0, p1, self.n_input):
                self.B[l:l+self.n_state, m:m+self.n_input] = B_stack[:,p2:p2+self.n_input]
                p2 += self.n_input
            p1 += self.n_input
        
    def compute_control(self, x_current):
        """
        Compute optimal control using quadratic programming
        x0: current state error [X; X_dot]
        """
        # Define optimization variables
        U = cp.Variable((self.n_input*(self.N), 1))

        # Define Cost Function
        x0 = x_current
        X = self.A @ x0 + self.B @ U
        
        X_ref = self.x_ref[:, 1+self.step:1+self.step+self.N].reshape(-1, 1)
        E_cost = cp.quad_form(X - X_ref, self.Q)

        U_cost = cp.quad_form(U, self.R)

        cost = E_cost + U_cost

        prob = cp.Problem(cp.Minimize(cost))

        try:
            # Set verbose to True to get more information about solver
            prob.solve(verbose=False)
        except Exception as e:
            print(f"Optimization error: {e}")
            raise

        self.step += 1

        if prob.status == cp.OPTIMAL:
            # Extract optimal control input 
            u_optimal = U.value[:self.n_input]
            print(Fore.GREEN + f"Optimization Status: {prob.status}")
            print(Style.RESET_ALL)
            return u_optimal
        
        elif prob.status == cp.INFEASIBLE:
            print(Fore.RED + "Optimization problem is INFEASIBLE")
            print(Style.RESET_ALL)
            return np.zeros((self.n_input,1))
        
        else:
            print(f"Unexpected problem status: {prob.status}")
            raise ValueError(f"Optimization problem: {prob.status}")
