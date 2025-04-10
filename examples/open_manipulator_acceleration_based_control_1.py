import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *
import scipy.interpolate
import scipy.linalg as LA
from scipy.signal import savgol_filter

# from my_DMP.dmp import DMP
from robot_model import Robot_Dynamics
from controllers import Controller

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

COG_wrt_body = [Matrix([[0],[0],[0]]), Matrix([[0.106],[-0.014],[0]]), Matrix([[0.0933],[0],[0]]), Matrix([[0.06047],[0],[0]])]  # location of COG wrt to DH-frame / body frame
MOI_about_body_CG = []  # MOI of the link about COG
# for i in range(len(mass)):
    # MOI_about_body_CG.append(Matrix([[0,  0,  0],
    #                                  [0,  0,  0],
    #                                  [0,  0,  0]]))

joint_limits = {'upper': np.radians([180, 90, 87.5, 114.5]),
                'lower': np.radians([-180, -117, -90, -103]),
                'vel_max': np.array([2.0, 2.0, 2.0, 2.0]),  # Maximum joint velocities (180deg/s)
                }

# if you change any kinematic or dynamic parameters then delete the saved .pkl model and re-create the model 
robot = Robot_Dynamics(kinematic_property, mass, COG_wrt_body, MOI_about_body_CG, joint_limits=joint_limits, file_name="Open_X_manipulator")
controller = Controller(robot)

# Robot Initial State (Joint Space)
q = np.radians([60.0, 102.0, -108.0, -40.0])  # In radian
q_dot = np.array([0, 0, 0, 0])  # In radian/sec
q_ddot = np.array([0, 0, 0, 0])  # In radian/sec2

# Control Gain
Kp_ts = np.diag([20,20,20])  # Proportional gains for joint space control
Kd_ts = 1.5*np.sqrt(Kp_ts)   # Derivative gains for joint space control

Kd_js = np.diag([5,5,5,5])     # Derivative gains for joint space control

""" Trajectory tracking """
# trajectory 
data = np.load('../data/'+'/'+str('example')+'.npz')
x_des = data['demo'].T  # (2,611)
x_des /= 350 # convert data from mm to m

# add Z-axis to data
x_des = np.vstack((x_des, np.linspace(0.1, 0.18, x_des.shape[1])))  # (3,611)
x_des[0,:] += 0.2
x_des[1,:] += 0.1

t = np.linspace(0, 5, x_des.shape[1])  # demo trajectory timing
dt = t[1] - t[0]

pos_interp = scipy.interpolate.interp1d(t, x_des, axis=1)
Xd = pos_interp(t)

# Apply Savitzky-Golay filter to each dimension
window_length = 51  # Must be odd; adjust based on your data
poly_order = 3  # Adjust based on your data
Xd_dot = np.array([savgol_filter(Xd[i], window_length, poly_order, deriv=1, delta=dt) for i in range(Xd.shape[0])])

# Calculate acceleration
Xd_ddot = np.array([savgol_filter(Xd_dot[i], window_length, poly_order, deriv=1, delta=dt) for i in range(Xd_dot.shape[0])])

# Start plotting tool
robot.plot_start(dt, t)

# Robot Initial State in Task-Space
robot.initial_state(q)

"""This formulation is from `Operational Space Control: A Theoretical and Empirical Comparison` section 3.2.2"""
# Simulation loop
for i in range(Xd.shape[1]-1):   
    # Kinematic Model
    Xe, Xe_dot, _ = robot.robot_KM.FK(q, q_dot, q_ddot)
    
    # task space error
    Ex = Xd[:,[i]] - Xe
    Ex_dot = Xd_dot[:,[i]] - Xe_dot

    # Kinematic Control
    qr, qr_ddot = controller.KC.acceleration_based_control_1(q, q_dot, Ex, Ex_dot, Xd_ddot[:,[i]], Kp_ts, Kd_ts, Kd_js)

    # Torque Control
    tau = controller.torque_control_1(q, q_dot, qr_ddot)

    # Joint space error
    Er = (qr - q).reshape((n, 1)) 

    X_cord, Y_cord, Z_cord = robot.robot_KM.taskspace_coord(q)
    robot.memory(X_cord, Y_cord, Z_cord, Er, tau, Xd[:,[i]], Ex, Ex_dot)

    # Robot Joint acceleration
    q, q_dot, q_ddot = robot.forward_dynamics(q, q_dot, tau, forward_int="euler_forward")  # forward_int = None / euler_forward / rk4
    
robot.show_plot_taux()