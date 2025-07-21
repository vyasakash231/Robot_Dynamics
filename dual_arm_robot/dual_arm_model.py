import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *

from robot_model import Robot_Dynamics
from controllers import Controller


# --------------------------------------------- Open Manipulator -------------------------------------------- #

# DH-Parameters
n = 6  # DOF (No of Joint)
alpha = np.radians([0, -90, 0, 90, -90, 90])  # In radians
a = np.array([0, 0, 0.409, 0, 0, 0])  # in meters
d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])  # in meters
d_nn = Matrix([[0.0], [0.0], [0.15]])  # coord of EE wrt to last joint frame in meters
gravity_axis = Matrix([[0], [0], [1]])  # gravity acting along Z0-axis

kinematic_property = {'dof':n, 'alpha':alpha, 'a':a, 'd':d, 'd_nn':d_nn}

# Dyanamic ParametersX_cord
mass = np.array([3.72, 6.84, 2.77, 2.68, 2.05, 0.87])  # in kg

COG_wrt_body = [Matrix([[-0.000069], [0.0244230], [-0.007375]]), 
                Matrix([[0.2044860], [0.0000000], [0.1327100]]), 
                Matrix([[0.0000710], [-0.005123], [0.0330100]]), 
                Matrix([[-0.000086], [0.0863480], [-0.066031]]), 
                Matrix([[0.0000910], [-0.006457], [0.0154340]]), 
                Matrix([[-0.000022], [-0.000007], [-0.050746]])]  # location of COG wrt to DH-frame / body frame

MOI_about_body_CG = []  # MOI of the link about COG
for i in range(len(mass)):
    MOI_about_body_CG.append(Matrix([[0,  0,  0],
                                     [0,  0,  0],
                                     [0,  0,  0]]))

joint_limits = {'upper': np.radians([360, 95, 135, 360, 135, 360]),
                'lower': np.radians([-360, -95, -135, -360, -135, -360]),
                'vel_max': np.radians([90, 90, 90, 90, 90, 90])
                }

# if you change any kinematic or dynamic parameters then delete the saved .pkl model and re-create the model 
robot = Robot_Dynamics(kinematic_property, mass, COG_wrt_body, MOI_about_body_CG, gravity_axis=gravity_axis, joint_limits=joint_limits, file_name="Doosan_A0509S")
# controller = Controller(robot)

# # Robot Initial State (Joint Space)
# q = np.array([0.93028432, 1.78183731, -1.8493209, -0.78539816])  # In radian
# q_dot = np.array([0, 0, 0, 0])  # In radian/sec
# q_ddot = np.array([0, 0, 0, 0])  # In radian/sec2

# # Impedence Control Gain
# Kd = np.diag([10, 10, 10])   # Stiffness matrix
# Dd = 2*np.sqrt(Kd)   # Damping matrix

# Xd = np.array([[0.15], [0.05], [0.18]])
# dt = 0.01

# # # Start plotting tool
# # robot.plot_start(dt)

# # Robot Initial State in Task-Space
# robot.initial_state(q)

# Er = None # Joint space error
# i = 0

# """This formulation essentially allows you to control the joint torques in a manner that tracks the referance joint positions, velocities"""
# # Simulation loop
# while True:   
#     external_force = np.array([[0], [0], [0]])     
    
#     # Kinematic Model
#     Xe, Xe_dot, _ = robot.robot_KM.FK(q, q_dot, q_ddot)
    
#     # task space error
#     Ex = Xe - Xd

#     # ---------- WRITE CONTROLLER -----------

#     X_cord, Y_cord, Z_cord = robot.robot_KM.taskspace_coord(q)

#     robot.memory(X_cord, Y_cord, Z_cord, Er, tau, Xd, Ex, None, external_force)

#     # Robot Joint acceleration
#     q, q_dot, q_ddot = robot.forward_dynamics(q, q_dot, tau, forward_int="euler_forward", ext_force=external_force)  # forward_int = None / euler_forward / rk4

#     i += 1

#     if i == 700:
#         break
    
