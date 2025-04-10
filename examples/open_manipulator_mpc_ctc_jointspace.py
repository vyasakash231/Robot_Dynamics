import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *
import scipy.interpolate
import scipy.linalg

# from my_DMP.dmp import DMP
from robot_model import Robot_Dynamics
from controllers import Controller
from utils import *

def smooth_velocity(pos, t, smoothing_factor=1.0):
    spline = scipy.interpolate.UnivariateSpline(t, pos, s=smoothing_factor)
    return spline.derivative()(t)

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

# Robot Goal State
q_goal = np.radians([90.0, 45.0, -45.0, 45.0])

# Generate Joint Referance Trajectory
N = 25  # prediction horizon

data = np.load('../data/'+'/'+str('example_q')+'.npz')
qd, qd_dot, qd_ddot = data["q"].T, data["qd_dot"].T, data["qd_ddot"].T

T = 5
t = np.linspace(0, T, qd.shape[1])  # demo trajectory timing
dt = t[1] - t[0]

Q_ref = np.block([[qd], [qd_dot]])
last_column = Q_ref[:,[-1]]
repeat_column = np.repeat(last_column, N, axis=1)
Q_ref = np.hstack((Q_ref, repeat_column))  # repeat last column n*N times

# Start plotting tool
robot.plot_start(dt, t)

# Robot Initial State in Joint-Space
# q = np.radians([45.0, 90.0, -90.0, -45.0])  # In radian
q = np.radians([60.0, 102.0, -108.0, -40.0])  # In radian
q_dot = np.array([0, 0, 0, 0])  # In radian/sec
robot.initial_state(q)

# Start MPC
controller.start_jointspace_mpc(Q_ref, N, state_cost=500.0, input_cost=2.0)

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