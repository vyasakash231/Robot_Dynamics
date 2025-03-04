import os
import sys
import numpy as np
import numpy.linalg as LA
np.set_printoptions(suppress=True)
from sympy import *
import scipy.interpolate
import scipy.linalg

# from my_DMP.dmp import DMP
from robot_model import Robot_Dynamics
from controllers import Controller


def smooth_velocity(pos, t, smoothing_factor=1.0):
    spline = scipy.interpolate.UnivariateSpline(t, pos, s=smoothing_factor)
    return spline.derivative()(t)


# --------------------------------------------- Open Manipulator -------------------------------------------- #
"""https://studywolf.wordpress.com/2013/12/05/dynamic-movement-primitives-part-2-controlling-a-system-and-comparison-with-direct-trajectory-control/"""

# Define obstacles
obstacle_def = [{"center": np.array([0.24, 0.05, 0.125]), 'ref_point': np.array([0.01, 0.01, 0.01]), 'radius': np.array([0.03, 0.03, 0.03]), "order": np.array([2, 2, 2]), "eta": 1.05, "rho": 1.0, "color": [0.25, 0.25, 0.25], 'margin': 0.1},
            ]


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
robot = Robot_Dynamics(kinematic_property, mass, COG_wrt_body, MOI_about_body_CG, 
                       joint_limits=joint_limits, obstacles=obstacle_def, method="DS",   # method = DS / APF / None
                       file_name="Open_X_manipulator")
controller = Controller(robot)  

# Robot Initial State (Joint Space)
q = np.radians([53.0, 102.0, -106.0, -45.0])  # In radian
q_dot = np.array([0, 0, 0, 0])  # In radian/sec
q_ddot = np.array([0, 0, 0, 0])  # In radian/sec2

# Robot Goal State
q_goal = np.radians([90.0, 45.0, -45.0, 45.0])

""" Trajectory tracking """
# trajectory 
data = np.load('../data/'+'/'+str('example')+'.npz')
x_des = data['demo'].T  # (2,611)
x_des /= 350 # convert data from mm to m

# add Z-axis to data
x_des = np.vstack((x_des, np.linspace(0.1, 0.2, x_des.shape[1])))  # (3,611)
x_des[0,:] += 0.2
x_des[1,:] += 0.1

T = 4
t = np.linspace(0, T, x_des.shape[1])  # demo trajectory timing
dt = t[1] - t[0]

pos_interp = scipy.interpolate.interp1d(t, x_des, axis=1)
Xd = pos_interp(t)

Xd_dot = np.array([smooth_velocity(Xd[i], t) for i in range(Xd.shape[0])])

# gain matrix
Kp_ts = np.diag([50, 50, 50])
Kd_ts = np.diag([8, 8, 8])
Kp_js = np.diag([3, 3, 3, 3]) 

# Start plotting tool
robot.plot_start(dt, t)

# Robot Initial State in Task-Space
robot.robot_KM.initial_state(q)

# Start DMPS
controller.start_dmp(no_of_DMPs=Xd.shape[0], no_of_basis=30, run_time=T, K=100, alpha=3.0)

# learn Weights based on Demo
controller.dmp.learn_dynamics(X_des=Xd)
controller.dmp.reset_state()

gamma = 1
# Simulation loop
for i in range(t.shape[0]-1):
    # Kinematic Model
    Xe, Xe_dot, _ = robot.robot_KM.FK(q, q_dot, q_ddot)

    # task space error
    Ex = Xd[:,[i]] - Xe
    Ex_dot = Xd_dot[:,[i]] - Xe_dot

    # DMP step
    # Xdmp, Xdmp_dot = controller.dmp.step(Xd[:,[-1]], gamma)
    Xdmp, Xdmp_dot = controller.dmp.step_with_DS_2012(Xd[:,[-1]], gamma)

    # Kinematic Control
    Xr_ddot = controller.KC.pd_dmp(Xdmp, Xdmp_dot, Xe, Xe_dot, Kp_ts, Kd_ts)

    # Feed-forward Control
    # tau = controller.torque_control_2(Xr_ddot, q, q_dot)   # without null-space torque
    tau = controller.torque_control_3(Xr_ddot, q, q_dot, Kp_js)    # with null-space torque

    # Forward Kinematics
    X_cord, Y_cord, Z_cord = robot.robot_KM.taskspace_coord(q)
    robot.memory(X_cord, Y_cord, Z_cord, None, tau, Xd[:,[i]], Ex, Ex_dot, None, Xdmp)
    
    """
    If the plant/Robot state drifts away from the state of the DMPs, we have to slow down the execution speed of the 
    DMP to allow the plant time to catch up. To do this we just have to multiply the DMP timestep dt with gamma
    """
    gamma = 1 / (1 + LA.norm(Xdmp - Xe))

    # Robot Joint acceleration
    q, q_dot, q_ddot = robot.forward_dynamics(q, q_dot, tau, forward_int="euler_forward")  # forward_int = None / euler_forward / rk4  
robot.show_plot_dmp()
