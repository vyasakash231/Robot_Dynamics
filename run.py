from math import *
import numpy as np
np.set_printoptions(suppress=True)
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from euler_lagrange import Euler_Lagrange

# l0, l1, m0, m1 = Symbol("l0"), Symbol("l1"), Symbol("m0"), Symbol("m1")

# Time array
t = np.linspace(0, 20, 2500)

# DH-Parameters
n = 2  # DOF 
# l = [l0, l1]
l = np.array([1, 1])
alpha = np.radians([0, 0])
a = np.array([0, l[0]])
d = np.array([0, 0])

d_nn = Matrix([l[1], 0, 0])  # coord of EE wrt to last joint frame

# Dyanamic ParametersX_cord
m = np.array([1, 1])
# m = [m0, m1]

COG_wrt_body = []  # location of COG wrt to DH-frame / body frame
MOI_about_body_CG = []  # MOI of the link about COG based on DH-frame / body frame
for i in range(len(m)):
    COG_wrt_body.append(Matrix([[l[i]],[0],[0]]))
    # MOI_about_body_CG.append(Matrix([[0,          0         ,          0         ],
    #                                  [0, (m[i]/12)*(l[i]**2),          0         ],
    #                                  [0,          0         , (m[i]/12)*(l[i]**2)]]))
    MOI_about_body_CG.append(Matrix([[0,  0,  0],
                                     [0,  0,  0],
                                     [0,  0,  0]]))

doosan = Euler_Lagrange(n, COG_wrt_body, MOI_about_body_CG, d_nn)

# Symbolic Matrices
M_sym, C_sym, G_sym = doosan.mcg_matrix(alpha, a, d, m)

# Save Matrices
# doosan.save_equations(M_sym, C_sym, G_sym)

# print(M_sym,"\n")
# print(C_sym,"\n")
# print(G_sym,"\n")

# Initial conditions
q = [0, 0]
q_dot = [0, 0]

# Solve ODE
doosan.free_fall_forward_dynamics(t, q, q_dot)

# Create the animation
anim = FuncAnimation(doosan.fig, doosan.update, frames=doosan.X_plot.shape[0], interval=30, blit=False, repeat=False)
plt.tight_layout()
plt.show()

# # Optional: Save the animation
# anim.save('double_pendulum.gif', writer='pillow', fps=30)