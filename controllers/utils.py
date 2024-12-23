import numpy as np


class Canonical_System:
    """
    τ * (dθ/dt) = - alpha * θ
    
    Intergrate both sides;
    intgrate(dθ/θ) = -(alpha/τ) * integrate(dt)
    ln(θ2) - ln(θ1) = -(alpha/τ) * (t2 - t1)

    ln(θ2/θ1) = -(alpha/τ) * (t2 - t1)
    θ2/θ1 = exp(-(alpha/τ) * dt)
    θ2 = exp(-(alpha/τ) * dt) * θ1
    """
    def __init__(self, dt, alpha, run_time = 1):
        self.dt = dt
        self.alpha = alpha
        self.run_time = run_time  # T
        self.time_steps = int(run_time/dt)  # T/dt = 1/0.005 = 200 time steps

        self.reset()

    def reset(self):
        """Reset the system state"""
        self.theta = 1  # at t = 0, theta = 1

    def step(self,tau=1):
        """Perform single step integration"""
        self.theta = np.exp(-(self.alpha/tau)*self.dt) * self.theta

    def rollout(self,tau=1):
        self.theta_track = np.zeros(self.time_steps)
        self.reset()
        for i in range(self.time_steps):
            self.theta_track[i] = self.theta
            self.step(tau)
        return self.theta_track
    
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def rk4_step(Y, A, B, dt):
    def f(Y):
        return A @ Y + B

    k1 = f(Y)
    k2 = f(Y + 0.5 * dt * k1)
    k3 = f(Y + 0.5 * dt * k2)
    k4 = f(Y + dt * k3)
    
    dY = (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return dY

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def forward_euler(Y, A, B, dt):
    def f(Y):
        return A @ Y + B

    return f(Y)