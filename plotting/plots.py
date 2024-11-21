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
        self.T = getattr(simulation, 'T', None)
        self.e_joint = getattr(simulation, 'joint_error_plot', None)
        self.tau_plot = getattr(simulation, 'tau_plot', None)
        self.Xd_plot = getattr(simulation, 'Xd_plot', None)
        self.e_task = getattr(simulation, 'position_error_plot', None)
        self.de_task = getattr(simulation, 'velocity_error_plot', None)
        self.ext_force = getattr(simulation, 'wrench', None)

        if self.T is None:
            self.T = np.linspace(0, 1, self.x_plot.shape[0]-1) 

        if self.e_joint is not None:
            self.e_joint = np.array(self.e_joint, dtype=np.float64).T
        if self.tau_plot is not None:
            self.tau_plot = np.array(self.tau_plot, dtype=np.float64).T
        if self.Xd_plot is not None:
            self.Xd_plot = np.array(self.Xd_plot, dtype=np.float64).T
        if self.e_task is not None:
            self.e_task = np.array(self.e_task, dtype=np.float64).T
        if self.de_task is not None:
            self.de_task = np.array(self.de_task, dtype=np.float64).T
        if self.ext_force is not None:
            self.ext_force = np.array(self.ext_force, dtype=np.float64)

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
        anim = FuncAnimation(self.fig, update_func, frames=frames, interval=50 if self.T is None else 10, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        return anim
    
    def update_free_fall(self, frame):
        k = frame
        self.ax.clear()
        self._plot_robot_state(self.ax, k)
        self._set_3d_plot_properties_pendulum(self.ax, 90, -90)   # set graph properties 
        return self.ax
    
    def update_computed_torque_in_joint_space(self, frame):
        k = frame
        
        # Robot visualization
        self._plot_robot_state(self.axs[0], k)
        self._plot_joint_errors(self.axs[1], k)  # Error plot
        self._plot_torques(self.axs[2], k)  # Torque plot
        self._set_3d_plot_properties(self.axs[0], 90, -90)  # set graph properties 
        return self.axs
    
    def update_computed_torque_in_task_space(self, frame):
        k = frame
        
        # Robot visualization
        self._plot_robot_state(self.axs[0], k)
        self._plot_joint_errors(self.axs[1], k)  # Error plot
        self._plot_torques(self.axs[2], k)  # Torque plot
        self._plot_position_errors(self.axs[3], k)  # Error plot
        self._plot_velocity_errors(self.axs[4], k)   # Error plot
        self._set_3d_plot_properties(self.axs[0], 9, -14)   # set graph properties 
        return self.axs

    def update_impedence_control_in_task_space(self, frame):
        k = frame
        
        # Robot visualization
        self._plot_robot_state(self.axs[0], k)
        self._plot_torques(self.axs[1], k)  # Torque plot
        self._plot_position_errors(self.axs[2], k)  # Torque plot
        self._set_3d_plot_properties(self.axs[0], 9, -14)   # set graph properties 
        return self.axs    

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    def _plot_robot_state(self, ax, k):
        ax.clear()
        try:
            # plot external wrench
            ax.quiver(self.x_plot[k, -1], self.y_plot[k, -1], self.z_plot[k, -1], self.ext_force[k, 0], self.ext_force[k, 1], 
                      self.ext_force[k, 2], length=0.05, normalize=True, linewidths=1.25, colors=[0,0,0])
        except:
            pass
        try:
            # Plot desired trajectory
            ax.plot(self.Xd_plot[0,:], self.Xd_plot[1,:], self.Xd_plot[2,:], '-r', linewidth=2)
        except:
            pass    
        for j in range(self.x_plot.shape[1]):
            ax.plot(self.x_plot[k,j:j+2], self.y_plot[k,j:j+2], self.z_plot[k,j:j+2], '-', linewidth=10-j)
            ax.plot(self.x_plot[k,j], self.y_plot[k,j], self.z_plot[k,j], 'ko', linewidth=10)
        
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
    def _set_3d_plot_properties_pendulum(self, ax, elev, azim):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
    
    def _set_3d_plot_properties(self, ax, elev, azim):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(np.min(self.x_plot) - 0.1, np.max(self.x_plot) + 0.1)
        ax.set_ylim(np.min(self.y_plot) - 0.1, np.max(self.y_plot) + 0.1)
        ax.set_zlim(np.min(self.z_plot), np.max(self.z_plot) + 0.1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

    def show(self):
        plt.show()