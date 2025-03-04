import numpy as np

def quantic_trajectory_vector(n, t_0, t_f, dt, q_0, q_f, v_0, v_f, alpha_0, alpha_f):
    # Matrix A remains the same
    A = np.array([[1, t_0, t_0**2, t_0**3, t_0**4, t_0**5],
                  [0, 1, 2*t_0, 3*t_0**2, 4*t_0**3, 5*t_0**4],
                  [0, 0, 2, 6*t_0, 12*t_0**2, 20*t_0**3],
                  [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5],
                  [0, 1, 2*t_f, 3*t_f**2, 4*t_f**3, 5*t_f**4],
                  [0, 0, 2, 6*t_f, 12*t_f**2, 20*t_f**3]])
    
    # Time array
    t = np.arange(t_0, t_f, dt)
    
    # Initialize arrays for position, velocity, and acceleration
    q = np.zeros((n, len(t)))
    dq = np.zeros((n, len(t)))
    ddq = np.zeros((n, len(t)))
    
    # Solve for each dimension independently
    for i in range(n):
        # Vector b for each dimension
        b = np.array([q_0[i], v_0[i], alpha_0[i], q_f[i], v_f[i], alpha_f[i]])
        
        # Solve Ax = b
        x = np.linalg.solve(A, b)
        
        # Compute position, velocity, and acceleration
        q[i, :] = x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3 + x[4]*t**4 + x[5]*t**5
        dq[i, :] = x[1] + 2*x[2]*t + 3*x[3]*t**2 + 4*x[4]*t**3 + 5*x[5]*t**4
        ddq[i, :] = 2*x[2] + 6*x[3]*t + 12*x[4]*t**2 + 20*x[5]*t**3
    return t, q, dq, ddq

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def resample(surface, num_points=20):
    # Calculate the total length of the original curve
    total_length = np.sum([distance(surface[i], surface[i+1]) for i in range(len(surface)-1)])

    # Calculate the spacing between points
    spacing = total_length / (num_points - 1)

    # Initialize variables for the new trajectory
    new_trajectory = [surface[0]]
    current_position = surface[0]
    remaining_distance = spacing

    # Iterate through the original curve to create the new resampled trajectory
    for point in surface[1:]:
        dist_to_next_point = distance(current_position, point)

        # Check if we've reached the desired spacing
        if remaining_distance <= dist_to_next_point:
            # Interpolate to find the new point
            t = remaining_distance / dist_to_next_point
            new_point = [
                current_position[0] + t * (point[0] - current_position[0]),
                current_position[1] + t * (point[1] - current_position[1])
            ]
            new_trajectory.append(new_point)
            current_position = new_point
            remaining_distance = spacing
        else:
            # Move to the next point
            current_position = point
            remaining_distance -= dist_to_next_point

    # Ensure that the new trajectory has the correct number of points
    while len(new_trajectory) < num_points:
        new_trajectory.append(surface[-1])

    # Convert the new trajectory to a numpy array
    new_trajectory = np.array(new_trajectory)
    return new_trajectory