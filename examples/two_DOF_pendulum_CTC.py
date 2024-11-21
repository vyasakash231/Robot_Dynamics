import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *
import scipy.interpolate
import scipy.linalg

# Add the parent directory 'PhD' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mathematical_model import Robot_Dynamics

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

# Desired Trajectory
a, b = 0.1, 0.1
q_des = np.array([a*np.sin(np.pi*t), b*np.cos(np.pi*t)])
q_dot_des = np.array([a*np.pi*np.cos(np.pi*t), -b*np.pi*np.sin(np.pi*t)])
q_ddot_des = np.array([-a*(np.pi**2)*np.sin(np.pi*t), -b*(np.pi**2)*np.cos(np.pi*t)])

# Control Gain
Kp = np.diag([100,100])
Kd = np.diag([20,20])

# Computed-Torque Control (with Feed-back PD-control Input)
robot.computed_torque_control(t, q_initial, q_dot_initial, q_des, q_dot_des, q_ddot_des, Kp, Kd)

# # Optional: Save the animation
# # robot.anim.save('double_pendulum.gif', writer='pillow', fps=30)

