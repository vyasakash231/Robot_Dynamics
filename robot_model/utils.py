import numpy as np


def apply_joint_limits(q, q_dot, q_limits, k_spring=100, b_damper=10):
    """Add spring-damper forces when near limits"""
    q_min, q_max = q_limits["lower"], q_limits["upper"]
    limit_force = np.zeros_like(q)
    
    for i in range(len(q)):
        if q[i] < q_min[i]:
            # Spring force + damping force when below minimum
            limit_force[i] = k_spring*(q_min[i] - q[i]) - b_damper*q_dot[i]
        elif q[i] > q_max[i]:
            # Spring force + damping force when above maximum
            limit_force[i] = k_spring*(q_max[i] - q[i]) - b_damper*q_dot[i]
    
    return limit_force

def enforce_joint_limits(q, q_dot, q_limits):
    """Directly clamp positions and velocities"""
    q_min, q_max = q_limits["lower"], q_limits["upper"]
    qd_min, qd_max = -q_limits["vel_max"], q_limits["vel_max"]
    
    # Clamp positions
    q_clamped = np.clip(q, q_min, q_max)
    
    # Clamp velocities
    qd_clamped = np.clip(q_dot, qd_min, qd_max)
    
    # Zero out velocities at position limits
    for i in range(len(q)):
        if (q[i] <= q_min[i] and q_dot[i] < 0) or (q[i] >= q_max[i] and q_dot[i] > 0):
            qd_clamped[i] = 0
            
    return q_clamped, qd_clamped