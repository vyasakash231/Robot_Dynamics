import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *

# from my_DMP.dmp import DMP
from robot_model import Robot_Dynamics
from controllers import Controller


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

COG_wrt_body = [Matrix([[0],[0],[0]]), Matrix([[0.106],[-0.014],[0]]), Matrix([[0.0933],[0],[0]]), Matrix([[0.06047],[0],[0]])]  # location of COG wrt to DH-frame / body frame
MOI_about_body_CG = []  # MOI of the link about COG
# for i in range(len(mass)):
    # MOI_about_body_CG.append(Matrix([[0,  0,  0],
    #                                  [0,  0,  0],
    #                                  [0,  0,  0]]))

# if you change any kinematic or dynamic parameters then delete the saved .pkl model and re-create the model 
robot = Robot_Dynamics(kinematic_property, mass, COG_wrt_body, MOI_about_body_CG, file_name="Open_X_manipulator")
controller = Controller(robot)

# Robot Initial State (Joint Space)
q = np.array([0.93028432, 1.78183731, -1.8493209, -0.78539816])  # In radian
q_dot = np.array([0, 0, 0, 0])  # In radian/sec
q_ddot = np.array([0, 0, 0, 0])  # In radian/sec2

# Impedence Control Gain
Kd = np.diag([10, 10, 10])   # Stiffness matrix
Dd = 2*np.sqrt(Kd)   # Damping matrix

Xd = np.array([[0.107163954125662], [0.143778624114491], [0.101078995781214]])
Xd_dot = np.array([[0], [0], [0]])
dt = 0.005

# Start plotting tool
robot.plot_start(dt)

# Robot Initial State in Task-Space
robot.robot_KM.initial_state(q)

Er = None # Joint space error
i = 0

"""This formulation essentially allows you to control the joint torques in a manner that tracks the referance joint positions, velocities"""
# Simulation loop
while True:   
    external_force = np.array([[0], [0], [0]])
    if i > 160 and i < 200:
        external_force = np.array([[0], [0], [2.4]])
    if i > 200 and i < 230:
        external_force = np.array([[0], [1.5], [0]])
    if i > 255 and i < 290:
        external_force = np.array([[1], [1], [1]])        
    
    # Kinematic Model
    Xe, Xe_dot, _ = robot.robot_KM.kinematic_model(dt, q, q_dot, q_ddot)
    
    # task space error
    Ex = Xe - Xd
    Ex_dot = Xe_dot - Xd_dot

    # Feed-forward Control
    tau = controller.impedence_control_static(q, q_dot, Ex, Ex_dot, Dd, Kd)

    X_cord, Y_cord, Z_cord = robot.robot_KM.FK(q)

    robot.memory(X_cord, Y_cord, Z_cord, Er, tau, Xd, Ex, Ex_dot, external_force)

    # Robot Joint acceleration
    q, q_dot, q_ddot = robot.forward_dynamics(q, q_dot, tau, forward_int="euler_forward", ext_force=external_force)  # forward_int = None / euler_forward / rk4

    i += 1

    if i == 700:
        break
    
robot.show_plot_impedence()