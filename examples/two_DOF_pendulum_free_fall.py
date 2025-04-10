import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *

from robot_model import Robot_Dynamics

# ----------------------------------------- 2-DOF Pendulum -------------------------------------------- #

# DH-Parameters
n = 2  # DOF 
l = np.array([1, 1])  # link length (Eucledian-distance btw joint)
alpha = np.radians([0, 0])
a = np.array([0, l[0]])
d = np.array([0, 0])
d_nn = np.array([[l[1]], [0], [0]])  # coord of EE wrt to last joint frame

kinematic_property = {'dof':n, 'alpha':alpha, 'a':a, 'd':d, 'd_nn':d_nn}

# Dyanamic ParametersX_cord
mass = np.array([1, 1])

COG_wrt_body = []  # location of COG wrt to DH-frame / body frame
MOI_about_body_CG = []  # MOI of the link about COG
for i in range(len(mass)):
    COG_wrt_body.append(Matrix([[l[i]],[0],[0]]))
    # MOI_about_body_CG.append(Matrix([[0,  0,  0],
    #                                  [0,  0,  0],
    #                                  [0,  0,  0]]))

robot = Robot_Dynamics(kinematic_property, mass, COG_wrt_body, MOI_about_body_CG, file_name="two_dof_pendulum")

# Initial Condition 
t = np.linspace(0, 5, 1000)
q_initial = np.array([0, 0])
q_dot_initial = np.array([0, 0])

# Start plotting tool
dt = t[1] - t[0]
robot.plot_start(dt, t)

# free-fall condition
robot.free_fall(q_initial, q_dot_initial)


