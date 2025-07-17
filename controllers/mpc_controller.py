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
    |x1(t+dt)|   |I dt*I| |x1(t)|   |(dt²/2)*I |                  |I  dt*I| 
    |        | = |      |*|     | + |          |*u(t), where Ad = |       |
    |x2(t+dt)|   |0    I| |x2(t)|   |  dt*I    |                  |0    I |
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
        
        self.Bd = np.block([[((self.h**2)/2) * np.eye(n, n)], 
                            [self.h * np.eye(n, n)]])  # (2nxn)
        
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


class Linear_MPC_JointTaskSpace:
    """
    Robot Linear Model: Double Integrator 
        q̈ = u (joint accelerations)
    
    State-space form:
    x = [x1; x2] = [q; q̇]  (joint positions and velocities)
    
    ẋ1 = q̇ = x2
    ẋ2 = q̈ = u

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

    therefore, the exact discrete-time system is;
    |x1(t+dt)|   |I dt*I| |x1(t)|   |(dt²/2)*I |                  |I  dt*I| 
    |        | = |      |*|     | + |          |*u(t), where Ad = |       |
    |x2(t+dt)|   |0    I| |x2(t)|   |   dt*I   |                  |0    I |

    Discrete-time system:
    x[k+1] = Ad * x[k] + Bd * u[k]
    
    Cost function includes:
    1. End-effector target reaching (task space)
    2. Joint velocity regularization
    3. Joint acceleration regularization
    """
    def __init__(self, n, N=20, dt=0.1, forward_kinematics_func=None):
        self.n = n  # number of joints
        self.dt = dt
        self.N = N   # prediction horizon
        
        # State and input dimensions
        self.n_state = 2 * n  # [q; q_dot]
        self.n_input = n      # joint accelerations

        # Forward kinematics function (user-provided)
        self.forward_kinematics = forward_kinematics_func
        
        # Discretize the continuous-time system using exact integration
        self.Ad = np.block([[np.eye(n, n), self.dt * np.eye(n, n)],
                            [np.zeros((n, n)), np.eye(n, n)]])  # (2n x 2n)
        
        self.Bd = np.block([[((self.dt**2)/2) * np.eye(n, n)], 
                            [self.dt * np.eye(n, n)]])  # (2n x n)
        
        # Cost weights (will be set by user)
        self.Q_task = None      # Task space cost weight
        self.Q_vel = None       # Joint velocity cost weight  
        self.R_accel = None     # Joint acceleration cost weight
        
        # Target end-effector position
        self.p_target = None
        
        # Current linearization point
        self.q_lin = None
        self.jacobian = None
        
    def set_cost_weights(self, Q_task=10.0, Q_vel=1.0, R_accel=0.1):
        """Set cost function weights"""
        if isinstance(Q_task, (int, float)):
            self.Q_task = Q_task * np.eye(3)  # Assuming 3D end-effector position
        else:
            self.Q_task = Q_task
            
        if isinstance(Q_vel, (int, float)):
            self.Q_vel = Q_vel * np.eye(self.n)
        else:
            self.Q_vel = Q_vel
            
        if isinstance(R_accel, (int, float)):
            self.R_accel = R_accel * np.eye(self.n)
        else:
            self.R_accel = R_accel
            
    def set_target(self, p_target):
        """Set target end-effector position"""
        self.p_target = p_target.reshape(-1, 1)
        
    def set_joint_limits(self, q_min, q_max, q_dot_max, u_max):
        """Set joint limits"""
        self.q_min = q_min
        self.q_max = q_max
        self.q_dot_min = -q_dot_max
        self.q_dot_max = q_dot_max
        self.u_min = -u_max
        self.u_max = u_max

        self._precompute_mpc_matrices()

    def _precompute_mpc_matrices(self):
        """
        Precompute prediction matrices for MPC:
        x[1] = Ad * x[0] + Bd * u[0]
        x[2] = Ad * x[1] + Bd * u[1]
        x[3] = Ad * x[2] + Bd * u[2]
         :     :     :     :     :

        now, reuse this equation;
        x[1] = Ad * x[0] + Bd * u[0]
        x[2] = Ad * (Ad * x[0] + Bd * u[0]) + Bd * u[1] => Ad^2 * x[0] + (Ad * Bd) * u[0] + Bd * u[1]
        x[3] = Ad * (Ad * (Ad * x[0] + Bd * u[0]) + Bd * u[1]) + Bd * u[2] => Ad^3 * x[0] + (Ad^2 * Bd) * u[0] + (Ad * Bd) * u[1] + Bd * u[2] 

        and so on, now we can write this is matrix form;

        |x[1]|   | Ad |          |   Bd            0            0            0        :    :    : |   | u[0] |
        |x[2]|   |Ad^2|          |  Ad*Bd          Bd           0            0        :    :    : |   | u[1] |
        |x[3]| = |Ad^3| * x[0] + | Ad^2*Bd        Ad*Bd         Bd           0        :    :    : | @ | u[2] |
        | :  |   | :  |          |    :             :           :            :        :    :    : |   |  :   |
        |x[N]|   |Ad^N|          |Ad^(N-1)*Bd  Ad^(N-2)*Bd  Ad^(N-3)*Bd  Ad^(N-4)*Bd ... Ad*Bd  Bd|   |u[N-1]|

        X[k+1|N] = A_prep * x[0] + B_prep @ U[k|N]
        """
        # Prediction matrices
        self.A_pred = np.zeros((self.N * self.n_state, self.n_state))
        self.B_pred = np.zeros((self.N * self.n_state, self.N * self.n_input))
        
        # Build A matrix (state prediction)
        for i in range(self.N):
            self.A_pred[i * self.n_state:(i + 1) * self.n_state, :] = np.linalg.matrix_power(self.Ad, i+1)

        # Build B matrix (input prediction) 
        for i in range(self.N):
            for j in range(i + 1):
                row_start = i * self.n_state
                row_end = (i + 1) * self.n_state
                col_start = j * self.n_input
                col_end = (j + 1) * self.n_input
                
                A_power = np.linalg.matrix_power(self.Ad, i - j)
                self.B_pred[row_start:row_end, col_start:col_end] = A_power @ self.Bd
        
    def _build_cost_matrices(self, x_current, J_current):
        """Build cost matrices for QP formulation"""
        q_current = x_current[:self.n].flatten()
        
        # Update Jacobian at current configuration
        self.jacobian = J_current
        self.q_lin = q_current
        
        # Current end-effector position
        if self.forward_kinematics is not None:
            p_current = self.forward_kinematics(q_current)
        else:
            p_current = np.zeros(3)  # Placeholder
            
        # Linearized task space error term
        e_tilde = self.p_target.flatten() - p_current + self.jacobian @ q_current
        
        # Selection matrices
        C_q = np.block([np.eye(self.n), np.zeros((self.n, self.n))])  # Extract q from x
        C_v = np.block([np.zeros((self.n, self.n)), np.eye(self.n)])  # Extract q_dot from x
        
        # Build cost matrices for entire horizon
        H_states = np.zeros((self.N * self.n_state, self.N * self.n_state))
        f_states = np.zeros((self.N * self.n_state, 1))
        H_inputs = np.zeros((self.N * self.n_input, self.N * self.n_input))
        
        for i in range(self.N):
            # State cost matrix for step i
            Q_combined = (C_q.T @ self.jacobian.T @ self.Q_task @ self.jacobian @ C_q + C_v.T @ self.Q_vel @ C_v)
            
            # Linear term for task space cost
            f_task = -C_q.T @ self.jacobian.T @ self.Q_task @ e_tilde.reshape(-1, 1)
            
            # Fill in block diagonal matrices
            state_idx = slice(i * self.n_state, (i + 1) * self.n_state)
            input_idx = slice(i * self.n_input, (i + 1) * self.n_input)
            
            H_states[state_idx, state_idx] = Q_combined
            f_states[state_idx, :] = f_task
            H_inputs[input_idx, input_idx] = self.R_accel
            
        return H_states, f_states, H_inputs

    def compute_control(self, x_current):
        """
        Compute optimal control using quadratic programming
        x_current: current state [q; q_dot]
        """
        if self.Q_task is None or self.Q_vel is None or self.R_accel is None:
            raise ValueError("Cost weights not set. Call set_cost_weights() first.")
            
        if self.p_target is None:
            raise ValueError("Target not set. Call set_target() first.")
        
        # Define optimization variables
        U = cp.Variable((self.n_input * self.N, 1))
        
        # Build cost matrices
        H_states, f_states, H_inputs = self._build_cost_matrices(x_current)
        
        # Predicted states
        X_pred = self.A_pred @ x_current + self.B_pred @ U
        
        # Cost function
        state_cost = cp.quad_form(X_pred, H_states) + f_states.T @ X_pred
        input_cost = cp.quad_form(U, H_inputs)
        
        cost = state_cost + input_cost
        
        # Constraints
        constraints = []
        
        # Joint limits
        for i in range(self.N):
            state_start = i * self.n_state
            input_start = i * self.n_input
            
            # Position limits: q_min <= q <= q_max
            for j in range(self.n):
                q_pred = (self.A_pred[state_start + j, :] @ x_current + self.B_pred[state_start + j, :] @ U)
                constraints.append(q_pred >= self.q_min[j])
                constraints.append(q_pred <= self.q_max[j])
                
                # Velocity limits: q_dot_min <= q_dot <= q_dot_max
                q_dot_pred = (self.A_pred[state_start + self.n + j, :] @ x_current + self.B_pred[state_start + self.n + j, :] @ U)
                constraints.append(q_dot_pred >= self.q_dot_min[j])
                constraints.append(q_dot_pred <= self.q_dot_max[j])
            
            # Acceleration limits: u_min <= u <= u_max
            for j in range(self.n_input):
                constraints.append(U[input_start + j] >= self.u_min[j])
                constraints.append(U[input_start + j] <= self.u_max[j])

        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(verbose=False)
        except Exception as e:
            print(f"Optimization error: {e}")
            raise

        if prob.status == cp.OPTIMAL:
            # Extract optimal control input (first time step)
            u_optimal = U.value[:self.n_input].reshape(-1, 1)
            print(Fore.GREEN + f"Optimization Status: {prob.status}, Cost: {prob.value:.4f}")
            print(Style.RESET_ALL)
            return u_optimal
        
        elif prob.status == cp.INFEASIBLE:
            print(Fore.RED + "Optimization problem is INFEASIBLE")
            print(Style.RESET_ALL)
            return np.zeros((self.n_input, 1))
        
        else:
            print(f"Unexpected problem status: {prob.status}")
            return np.zeros((self.n_input, 1))