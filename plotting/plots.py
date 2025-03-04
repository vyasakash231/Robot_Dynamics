import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

class RobotPlotter:
    def __init__(self, simulation):
        self.sim = simulation
        self.fig = None
        self.ax = None
        self.axs = None
        
        # Copy necessary attributes from simulation
        self.x_plot = np.array(simulation.X_plot, dtype=np.float64)
        self.y_plot = np.array(simulation.Y_plot, dtype=np.float64)
        self.z_plot = np.array(simulation.Z_plot, dtype=np.float64)
        self.n = simulation.n
        
        # These might be None for free_fall simulation
        self.obstacles = getattr(simulation, 'obstacles', None)
        self.T = getattr(simulation, 'T', None)
        self.e_joint = getattr(simulation, 'joint_error_plot', [])
        self.tau_plot = getattr(simulation, 'tau_plot', [])
        self.Xd_plot = getattr(simulation, 'Xd_plot', [])
        self.e_task = getattr(simulation, 'position_error_plot', [])
        self.de_task = getattr(simulation, 'velocity_error_plot', [])
        self.ext_force = getattr(simulation, 'wrench', [])
        self.Xdmp = getattr(simulation, 'dmp_traj', [])

        if self.T is None:
            self.T = np.linspace(0, 1, self.x_plot.shape[0]-1) 

        # if self.e_joint is not None:
        #     self.e_joint = np.array(self.e_joint, dtype=np.float64).T
        # if self.tau_plot is not None:
        #     self.tau_plot = np.array(self.tau_plot, dtype=np.float64).T
        # if self.Xd_plot is not None:
        #     self.Xd_plot = np.array(self.Xd_plot, dtype=np.float64).T
        # if self.e_task is not None:
        #     self.e_task = np.array(self.e_task, dtype=np.float64).T
        # if self.de_task is not None:
        #     self.de_task = np.array(self.de_task, dtype=np.float64).T
        # if self.ext_force is not None:
        #     self.ext_force = np.array(self.ext_force, dtype=np.float64)
        # if self.Xdmp is not None:
        #     self.Xdmp_plot = np.array(self.Xdmp, dtype=np.float64).T
        
        if  len(self.e_joint) != 0:
            self.e_joint = np.array(self.e_joint, dtype=np.float64).T
        if len(self.tau_plot) != 0:
            self.tau_plot = np.array(self.tau_plot, dtype=np.float64).T
        if len(self.Xd_plot) != 0:
            self.Xd_plot = np.array(self.Xd_plot, dtype=np.float64).T
        if len(self.e_task) != 0:
            self.e_task = np.array(self.e_task, dtype=np.float64).T
        if len(self.de_task) != 0:
            self.de_task = np.array(self.de_task, dtype=np.float64).T
        if len(self.ext_force) != 0:
            self.ext_force = np.array(self.ext_force, dtype=np.float64)
        if len(self.Xdmp) != 0:
            self.Xdmp_plot = np.array(self.Xdmp, dtype=np.float64).T
        
        # generate obstacle shape
        if self.obstacles is not None:
            self._generate_superquadric()
        
    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    def setup_free_fall_plot(self):
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        return self.create_animation(self.update_free_fall)
    
    def setup_computed_torque_in_joint_space(self):
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[:,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1]))
        self.axs.append(self.fig.add_subplot(gs[1,1]))
        return self.create_animation(self.update_computed_torque_in_joint_space)
    
    def setup_computed_torque_in_task_space(self):
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 4)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[:,:2], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,2]))
        self.axs.append(self.fig.add_subplot(gs[1,2]))
        self.axs.append(self.fig.add_subplot(gs[0,3]))
        self.axs.append(self.fig.add_subplot(gs[1,3]))
        return self.create_animation(self.update_computed_torque_in_task_space)
    
    def setup_task_space_dmp(self):
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 3)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[:,:2], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,2]))
        self.axs.append(self.fig.add_subplot(gs[1,2]))
        return self.create_animation(self.update_task_space_dmp)        
    
    def setup_impedence_control_in_task_space(self):
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[:,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1]))
        self.axs.append(self.fig.add_subplot(gs[1,1]))
        return self.create_animation(self.update_impedence_control_in_task_space)        
    
    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    def create_animation(self, update_func):
        frames = len(self.T)-1 if self.T is not None else self.x_plot.shape[0]-1
        anim = FuncAnimation(self.fig, update_func, frames=frames, interval=60 if self.T is None else 10, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        return anim
    
    def update_free_fall(self, frame):
        self.ax.clear()
        self._plot_robot(self.ax, frame)
        self._set_3d_plot_properties_robot(self.ax, 90, -90)   # set graph properties 
        return self.ax
    
    def update_computed_torque_in_joint_space(self, frame):
        # Robot visualization
        self._plot_robot(self.axs[0], frame)
        self._plot_joint_errors(self.axs[1], frame)  # Error plot
        self._plot_torques(self.axs[2], frame)  # Torque plot
        self._set_3d_plot_properties_robot(self.axs[0], 9, -15)  # set graph properties 
        return self.axs
    
    def update_computed_torque_in_task_space(self, frame):
        # Robot visualization
        self._plot_robot(self.axs[0], frame)
        self._plot_torques(self.axs[2], frame)  # Torque plot
        self._plot_position_errors(self.axs[3], frame)  # Error plot
        self._plot_velocity_errors(self.axs[4], frame)   # Error plot
        self._set_3d_plot_properties_robot(self.axs[0], 9, -25)   # set graph properties 
        return self.axs

    def update_task_space_dmp(self, frame):
        # Robot visualization
        self._plot_robot(self.axs[0], frame)
        self._plot_position_errors(self.axs[1], frame)  # Error plot
        self._plot_torques(self.axs[2], frame)  # Torque plot
        self._set_3d_plot_properties_robot(self.axs[0], 9, -25)
        return self.axs
      
    def update_impedence_control_in_task_space(self, frame):
        # Robot visualization
        self._plot_robot(self.axs[0], frame)
        self._plot_torques(self.axs[1], frame)  # Torque plot
        self._plot_position_errors(self.axs[2], frame)  # Torque plot
        self._set_3d_plot_properties_robot(self.axs[0], 9, -15)   # set graph properties 
        return self.axs    

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    def _plot_robot(self, ax, k):
        ax.clear()
        
        if len(self.ext_force) != 0:
            # plot external wrench
            ax.quiver(self.x_plot[k, -1], self.y_plot[k, -1], self.z_plot[k, -1], self.ext_force[k, 0], self.ext_force[k, 1], 
                      self.ext_force[k, 2], length=0.05, normalize=True, linewidths=1.25, colors=[0,0,0])
        
        if len(self.Xd_plot) != 0:
            # Plot desired trajectory
            ax.plot(self.Xd_plot[0,:], self.Xd_plot[1,:], self.Xd_plot[2,:], '-r', linewidth=2)
        
        # Plot obstacle
        if self.obstacles is not None:
            self._plot_multiple_obstacles(ax)  

        for j in range(self.x_plot.shape[1]):
            ax.plot(self.x_plot[k,j:j+2], self.y_plot[k,j:j+2], self.z_plot[k,j:j+2], '-', linewidth=10-j)
            ax.plot(self.x_plot[k,j], self.y_plot[k,j], self.z_plot[k,j], 'ko', markersize=10-j)
        
        # plot trajectory of EE
        ax.plot(self.x_plot[:k+1, -1], self.y_plot[:k+1, -1], self.z_plot[:k+1, -1], linewidth=1.5, color='b')

    def _plot_joint_errors(self, ax, k):
        ax.clear()
        ax.set_xlim(self.T[0], self.T[-1])
        ax.set_ylim(np.min(self.e_joint) - 0.01, np.max(self.e_joint) + 0.01)
        
        for i in range(self.n):
            ax.plot(np.linspace(self.T[0], self.T[k+1], k+1), self.e_joint[i, :k+1], label=f'e_r{i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Joint Angle Error (q_{r} - q) in radians')
    
    def _plot_torques(self, ax, k):
        ax.clear()
        ax.set_xlim(self.T[0], self.T[-1])
        ax.set_ylim(np.min(self.tau_plot) - 0.01, np.max(self.tau_plot) + 0.01)
        
        for i in range(self.n):
            ax.plot(np.linspace(self.T[0], self.T[k+1], k+1), self.tau_plot[i, :k+1], label=f'Joint_{i+1} torque')
        ax.set_xlabel('Time')
        ax.set_ylabel('Torque (Nm)')

    def _plot_position_errors(self, ax, k):
        ax.clear()
        ax.set_xlim(self.T[0], self.T[-1])
        ax.set_ylim(np.min(self.e_task) - 0.01, np.max(self.e_task) + 0.01)
        
        for i in range(3):
            ax.plot(np.linspace(self.T[0], self.T[k+1], k+1), self.e_task[i, :k+1], label=f'e_{i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position Error (m)')

    def _plot_velocity_errors(self, ax, k):
        ax.clear()
        ax.set_xlim(self.T[0], self.T[-1])
        ax.set_ylim(np.min(self.de_task) - 0.01, np.max(self.de_task) + 0.01)
        
        for i in range(3):
            ax.plot(np.linspace(self.T[0], self.T[k+1], k+1), self.de_task[i, :k+1], label=f'e_{i+1}_dot')
        ax.set_xlabel('Time')
        ax.set_ylabel('Velocity Error (m/s)')

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    def _generate_superquadric(self, num_points=100):
        self.X_obs = []
        self.Y_obs = []
        self.Z_obs = []

        for _, obstacle in enumerate(self.obstacles):
            xc, yc, zc = obstacle['center']
            r1, r2, r3 = np.array(obstacle['radius'])
            m1, m2, m3 = obstacle['order']

            # Create parametric angles
            eta = np.linspace(-np.pi/2, np.pi/2, num_points)
            omega = np.linspace(-np.pi, np.pi, num_points)
            ETA, OMEGA = np.meshgrid(eta, omega)
            
            # Superquadric equations in parametric form
            cos_eta = np.abs(np.cos(ETA))**(2/m1)
            sin_eta = np.abs(np.sin(ETA))**(2/m3)
            cos_omega = np.abs(np.cos(OMEGA))**(2/m1)
            sin_omega = np.abs(np.sin(OMEGA))**(2/m2)
            
            # Sign corrections
            cos_eta = cos_eta * np.sign(np.cos(ETA))
            sin_eta = sin_eta * np.sign(np.sin(ETA))
            cos_omega = cos_omega * np.sign(np.cos(OMEGA))
            sin_omega = sin_omega * np.sign(np.sin(OMEGA))
            
            # Generate the surface points
            X = r1 * cos_eta * cos_omega + xc
            Y = r2 * cos_eta * sin_omega + yc
            Z = r3 * sin_eta + zc

            self.X_obs.append(X)
            self.Y_obs.append(Y)
            self.Z_obs.append(Z)

    # def _generate_superquadric(self, num_points=100):
    #     self.X_obs = []
    #     self.Y_obs = []
    #     self.Z_obs = []

    #     for _, obstacle in enumerate(self.obstacles):
    #         xc, yc, zc = obstacle['center']
    #         r1, r2, r3 = np.array(obstacle['radius'])
    #         m1, m2, m3 = obstacle['order']

    #         # Create parametric angles
    #         phi = np.linspace(-np.pi, np.pi, num_points)
    #         theta = np.linspace(-np.pi/2, np.pi/2, num_points)
    #         phi, theta = np.meshgrid(phi, theta)
            
    #         r = (abs(np.cos(theta))**m1 * abs(np.cos(phi))**m1 +
    #             abs(np.cos(theta))**m2 * abs(np.sin(phi))**m2 +
    #             abs(np.sin(theta))**m1) ** (-1/min(m1, m2, m3))
            
    #         # Generate the surface points
    #         X = r * r1 * np.cos(theta) * np.cos(phi) + xc
    #         Y = r * r2 * np.cos(theta) * np.sin(phi) + yc
    #         Z = r * r3 * np.sin(theta) + zc

    #         self.X_obs.append(X)
    #         self.Y_obs.append(Y)
    #         self.Z_obs.append(Z)

    def _plot_multiple_obstacles(self, ax):
        for idx, obstacle in enumerate(self.obstacles):
            ax.plot_surface(self.X_obs[idx], self.Y_obs[idx], self.Z_obs[idx], color=obstacle['color'], alpha=0.8)

    def _set_3d_plot_properties_robot(self, ax, elev, azim):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-0.1, 0.3)
        ax.set_ylim(-0.1, 0.3)
        ax.set_zlim(-0.0, 0.4)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

    def show(self):
        plt.show()