import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *
import scipy.linalg

# from my_DMP.dmp import DMP
from robot_model import Robot_Dynamics
from controllers import Controller


def quantic_trajectory_vector(n, t_0, t_f, dt, q_0, q_f, v_0, v_f, alpha_0, alpha_f):
    # Matrix A remains the same
    A = np.array([[1, t_0, t_0**2, t_0**3, t_0**4, t_0**5],
                  [0, 1, 2*t_0, 3*t_0**2, 4*t_0**3, 5*t_0**4],
                  [0, 0, 2, 6*t_0, 12*t_0**2, 20*t_0**3],
                  [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5],
                  [0, 1, 2*t_f, 3*t_f**2, 4*t_f**3, 5*t_f**4],
                  [0, 0, 2, 6*t_f, 12*t_f**2, 20*t_f**3]])
    
    # Time array
    t = np.arange(t_0, t_f, dt)
    
    # Initialize arrays for position, velocity, and acceleration
    q = np.zeros((n, len(t)))
    dq = np.zeros((n, len(t)))
    ddq = np.zeros((n, len(t)))
    
    # Solve for each dimension independently
    for i in range(n):
        # Vector b for each dimension
        b = np.array([q_0[i], v_0[i], alpha_0[i], q_f[i], v_f[i], alpha_f[i]])
        
        # Solve Ax = b
        x = np.linalg.solve(A, b)
        
        # Compute position, velocity, and acceleration
        q[i, :] = x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3 + x[4]*t**4 + x[5]*t**5
        dq[i, :] = x[1] + 2*x[2]*t + 3*x[3]*t**2 + 4*x[4]*t**3 + 5*x[5]*t**4
        ddq[i, :] = 2*x[2] + 6*x[3]*t + 12*x[4]*t**2 + 20*x[5]*t**3
    return t, q, dq, ddq


# --------------------------------------------- Open Manipulator -------------------------------------------- #

# DH-Parameters
n = 4  # DOF (No of Joint)
alpha = np.radians([0,90,0,0])  # In radians
a = np.array([0,0,np.sqrt(0.128**2 + 0.024**2),0.124])  # in meters
d = np.array([0.077,0,0,0])  # in meters
d_nn = Matrix([[0.126], [0], [0]])  # coord of EE wrt to last joint frame in meters

kinematic_property = {'dof':n, 'alpha':alpha, 'a':a, 'd':d, 'd_nn':d_nn}

# Dyanamic ParametersX_cord
mass = np.array([0.0, 0.1423463, 0.13467049, 0.23550927])

COG_wrt_body = [Matrix([[0],[0],[0]]), 
                Matrix([[0.106],[-0.014],[0]]), 
                Matrix([[0.0933],[0],[0]]), 
                Matrix([[0.06047],[0],[0]])]  # location of COG wrt to DH-frame / body frame

MOI_about_body_CG = []  # MOI of the link about COG
# for i in range(len(mass)):
    # MOI_about_body_CG.append(Matrix([[0,  0,  0],
    #                                  [0,  0,  0],
    #                                  [0,  0,  0]]))

joint_limits = {'upper': np.radians([180, 90, 87.5, 114.5]),
                'lower': np.radians([-180, -117, -90, -103]),
                'vel_max': np.array([2.0, 2.0, 2.0, 2.0]),  # Maximum joint velocities (180 deg/s)
                }

# if you change any kinematic or dynamic parameters then delete the saved .pkl model and re-create the model 
robot = Robot_Dynamics(kinematic_property, mass, COG_wrt_body, MOI_about_body_CG, joint_limits=joint_limits, file_name="Open_X_manipulator")
controller = Controller(robot)

# Robot Initial State (Joint Space)
q0 = np.radians([53.0, 102.0, -106.0, -45.0])  # In radian
q0_dot = np.array([0, 0, 0, 0])  # In radian/sec
q0_ddot = np.array([0, 0, 0, 0])  # In radian/sec2

# Robot Goal State
q_goal = np.radians([90.0, 45.0, -45.0, 45.0])

# Generate Joint Referance Trajectory
N = 30  # prediction horizon
dt = 0.01  # sampling time
t, qd, qd_dot, qd_ddot = quantic_trajectory_vector(n, 0, 5.0, dt, q0, q_goal, q0_dot, q0_dot, q0_ddot, q0_ddot)
x_ref = np.block([[qd], [qd_dot]])
last_column = x_ref[:,[-1]]
repeat_column = np.repeat(last_column, N, axis=1)
x_ref = np.hstack((x_ref, repeat_column))  # repeat last column n*N times

# Start plotting tool
robot.plot_start(dt, t)

# Robot Initial State in Task-Space
q = np.radians([45.0, 90.0, -90.0, -45.0])  # In radian
q_dot = np.array([0, 0, 0, 0])  # In radian/sec
robot.robot_KM.initial_state(q)

# Start MPC
controller.start_jointspace_mpc(x_ref, N, state_cost=100.0, input_cost=2.0)

"""This formulation essentially allows you to control the joint torques 
in a manner that tracks the referance joint positions, velocities"""
# Simulation loop
for i in range(t.shape[0]-1):   
    # Feed-forward Control
    tau = controller.jointspace_mpc_ctc(q, q_dot, qd_ddot[:,i])

    E_joint = (q - qd[:,i])[:,np.newaxis]
    
    # Forward Kinematics
    X_cord, Y_cord, Z_cord = robot.robot_KM.taskspace_coord(q)
    robot.memory(X_cord, Y_cord, Z_cord, E_joint, tau)

    # Robot Joint acceleration
    q, q_dot, q_ddot = robot.forward_dynamics(q, q_dot, tau, forward_int="euler_forward")  # forward_int = None / euler_forward / rk4
    
robot.show_plot_tauj()