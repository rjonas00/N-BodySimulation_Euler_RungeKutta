import numpy as np
# Constants
M = 100  # Total mass
A = 0.5  # Half-mass radius
G_const = 0.430

# Velocity dispersion
def sigma_squared(r):
    return (M * G_const) / (6 * A) * (1 + r**2 / A**2)**(-1/2)


#returns r values based on the p(r) distribution function
def sample_r(num):
    q = np.random.uniform(0, 1, num)
    r = A * np.sqrt((1 - q)**(-2/3) - 1)
    return r

def generate_velocities(pos,cartesian_coordinates):
    v_x = []
    v_y = []
    v_z = []
    for i in range(len(pos)):
        v_r = np.random.normal(0, np.sqrt(sigma_squared(pos[i])))
        v_phi = np.random.normal(0, np.sqrt(sigma_squared(pos[i])))
        v_theta = np.random.normal(0, np.sqrt(sigma_squared(pos[i])))
        
        # Convert spherical velocities to Cartesian
        theta = np.arccos(cartesian_coordinates[i][2] / pos[i]) if pos[i] != 0 else 0
        phi = np.arctan2(cartesian_coordinates[i][1], cartesian_coordinates[i][0])
        
        v_x.append(v_r * np.sin(theta) * np.cos(phi) + v_phi * np.cos(theta) * np.cos(phi) - v_theta * np.sin(phi))
        v_y.append(v_r * np.sin(theta) * np.sin(phi) + v_phi * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi))
        v_z.append(v_r * np.cos(theta) - v_theta * np.sin(theta))
    
    return np.array(list(zip(v_x, v_y, v_z)))

def cartesian_coordinates(r):
    x = []
    y = []
    z = []
    for i in range(len(r)):
        theta = np.arccos(1-2 * np.random.uniform(0, 1))  
        phi = np.random.uniform(0, 2 * np.pi)  
        x.append(r[i] * np.sin(theta) * np.cos(phi))
        y.append(r[i] * np.sin(theta) * np.sin(phi))
        z.append(r[i] * np.cos(theta))
    return np.array(list(zip(x, y, z)))
