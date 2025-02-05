import numpy as np
import time as tm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
#used units in the nbody sim:
#  speed    = km/s
#  mass     = M_sun
#  distance = pc
#  time     = Myr

G_const = 0.430
a1      = 1.022
a2      = 1.022
v1      = 1.022
soft    = 1e-4

#calculation of the kinetic energy for all particles 
def calc_EK(v,m):
    aux_ek = 0.5*(m)*((v[:,0])**2+(v[:,1])**2+(v[:,2])**2)
    return np.sum(aux_ek)


#calculation of the potential energy for all particles
def calc_EU(x,m,G=G_const,epsilon = soft):

    mtrx_X_01 = np.tile(x[:,0],(len(x),1)).T
    mtrx_X_02 = np.tile(x[:,0],(len(x),1))

    mtrx_Y_01 = np.tile(x[:,1],(len(x),1)).T
    mtrx_Y_02 = np.tile(x[:,1],(len(x),1))

    mtrx_Z_01 = np.tile(x[:,2],(len(x),1)).T
    mtrx_Z_02 = np.tile(x[:,2],(len(x),1))

    mtrx_X = mtrx_X_01-mtrx_X_02
    mtrx_Y = mtrx_Y_01-mtrx_Y_02
    mtrx_Z = mtrx_Z_01-mtrx_Z_02

    mtrx_R = np.sqrt(mtrx_X**2 + mtrx_Y**2 + mtrx_Z**2 + epsilon )

    aux_rc = 1./(mtrx_R + np.identity(m.size)) - np.identity(m.size)

    aux_ms = np.tile(m,(m.size,1))*G

    return -0.5*np.sum(m*np.sum(aux_ms*aux_rc,axis=1))

#tracking of center of mass position and velocity due to drift by the numerical errors
def cm_pos(x,m):
    CMx = np.sum(x[:,0]*m)/(np.sum(m))
    CMy = np.sum(x[:,1]*m)/(np.sum(m))
    CMz = np.sum(x[:,2]*m)/(np.sum(m))

    return [CMx,CMy,CMz]

def cm_vel(v,m):
    CMvx = np.sum(v[:,0]*m)/(np.sum(m))
    CMvy = np.sum(v[:,1]*m)/(np.sum(m))
    CMvz = np.sum(v[:,2]*m)/(np.sum(m))

    return [CMvx,CMvy,CMvz]

def update(x0,v0,masses):
    cmX0 = cm_pos(x0,masses)
    cmV0 = cm_vel(v0,masses)

    #update initial conditions
    x0[:,0] = x0[:,0]-cmX0[0]
    x0[:,1] = x0[:,1]-cmX0[1]
    x0[:,2] = x0[:,2]-cmX0[2]

    v0[:,0] = v0[:,0]-cmV0[0]
    v0[:,1] = v0[:,1]-cmV0[1]
    v0[:,2] = v0[:,2]-cmV0[2]
    return x0,v0



#we get returned the gravitational force on each particle in all 3 directions at a current constellation of x
def calc_force(x,m,G=G_const,epsilon = soft):

    mtrx_X_01 = np.tile(x[:,0],(len(x),1)).T
    mtrx_X_02 = np.tile(x[:,0],(len(x),1))

    mtrx_Y_01 = np.tile(x[:,1],(len(x),1)).T
    mtrx_Y_02 = np.tile(x[:,1],(len(x),1))

    mtrx_Z_01 = np.tile(x[:,2],(len(x),1)).T
    mtrx_Z_02 = np.tile(x[:,2],(len(x),1))

    mtrx_X = (mtrx_X_01-mtrx_X_02)
    mtrx_Y = (mtrx_Y_01-mtrx_Y_02)
    mtrx_Z = (mtrx_Z_01-mtrx_Z_02)

    mtrx_R = np.sqrt( mtrx_X**2 + mtrx_Y**2 + mtrx_Z**2 + epsilon )

    aux_xc = (1./(mtrx_R + np.identity(len(mtrx_R)))**(3.) - np.identity(len(mtrx_R)))*mtrx_X
    aux_yc = (1./(mtrx_R + np.identity(len(mtrx_R)))**(3.) - np.identity(len(mtrx_R)))*mtrx_Y
    aux_zc = (1./(mtrx_R + np.identity(len(mtrx_R)))**(3.) - np.identity(len(mtrx_R)))*mtrx_Z

    aux_ms = np.tile(m,(m.size,1))*G

    GF_X = -np.sum(aux_ms*aux_xc,axis=1)
    GF_Y = -np.sum(aux_ms*aux_yc,axis=1)
    GF_Z = -np.sum(aux_ms*aux_zc,axis=1)

    return GF_X,GF_Y,GF_Z


def next_step_euler(x,v,m,dt):
    ###
    # x is the original position vector [x,y,z]
    # v is the original velocity vector [vx,vy,vz]
    # m is the mass array (mass of each n-th particles)
    # dt is the time step
    ###

    #First calculate the accelerations:
    ax,ay,az = calc_force(x,m)

    #velocity update "kick":
    vx_new = v[:,0] + (a1*ax)*(dt)
    vy_new = v[:,1] + (a1*ay)*(dt)
    vz_new = v[:,2] + (a1*az)*(dt)

    #position update "drift":
    xx_new = x[:,0] + (v1*v[:,0])*(dt) + 0.5*(a2*ax)*(dt)**2
    xy_new = x[:,1] + (v1*v[:,1])*(dt) + 0.5*(a2*ay)*(dt)**2
    xz_new = x[:,2] + (v1*v[:,2])*(dt) + 0.5*(a2*az)*(dt)**2

    x_new = np.array([xx_new,xy_new,xz_new]).T
    v_new = np.array([vx_new,vy_new,vz_new]).T

    return [x_new, v_new]

def derivative_acceleration(x, v, m):
    ax, ay, az = calc_force(x, m)
    dvdt = np.array([ax, ay, az]).T
    return dvdt

def derivative_velocity(x, v, m):
    dxdt = v
    return dxdt



def Runge_Kutta4(x, v, m, dt):
    #Calculate K1
    k_x_1=derivative_velocity(x, v, m) 
    k_v_1 = derivative_acceleration(x, v, m)
    
    x1 = x + 0.5 * dt * k_x_1
    v1 = v + 0.5 * dt * k_v_1
    
    #Calculate K2
    k_x_2=derivative_velocity(x1, v1, m) 
    k_v_2 = derivative_acceleration(x1, v1, m)
    
    x2 = x + 0.5 * dt * k_x_2
    v2 = v + 0.5 * dt * k_v_2
    
    #Calculate K3
    k_x_3=derivative_velocity(x2, v2, m) 
    k_v_3 = derivative_acceleration(x2, v2, m)

    x3 = x + dt * k_x_3
    v3 = v + dt * k_v_3
    
    #Calculate K4
    k_x_4=derivative_velocity(x3, v3, m) 
    k_v_4 = derivative_acceleration(x3, v3, m)
    
    #xnew and vnew calculation
    x_new = x + (dt / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)
    v_new = v + (dt / 6.0) * (k_v_1 + 2 * k_v_2 + 2 * k_v_3 + k_v_4)
    
    return x_new, v_new

def Energyloss_plotter(cartesian_pos,cartesian_vel,masses,t_total=50,t_resol=[100],typ="Euler"):
    # Total time = In Myrs

    t_steps = [1 / t_resol[i] for i in range(len(t_resol))]

    Nframes = [int(t_total / t_steps[i]) for i in range(len(t_resol))]

    t0 = tm.time()


    for i in range(len(Nframes)):
        # Reset for each resolution
        SNP_XV = [np.array([cartesian_pos, cartesian_vel])]
        SNP_EK = [calc_EK(SNP_XV[0][1], masses)]
        SNP_EU = [calc_EU(SNP_XV[0][0], masses)]

        for k in range(1, Nframes[i]):
            if typ=="RungeKutta":
                SNP_NEXT = Runge_Kutta4(SNP_XV[k-1][0], SNP_XV[k-1][1], masses, t_steps[i])
            else:
                SNP_NEXT = next_step_euler(SNP_XV[k-1][0], SNP_XV[k-1][1], masses, t_steps[i])
            SNP_EK.append(calc_EK(SNP_NEXT[1], masses))
            SNP_EU.append(calc_EU(SNP_NEXT[0], masses))
            SNP_XV.append(SNP_NEXT)

        # Convert lists to NumPy arrays
        SNP_XV = np.array(SNP_XV)
        SNP_EK = np.array(SNP_EK)
        SNP_EU = np.array(SNP_EU)

        print(f'Simulation for resolution {t_resol[i]} done in {(tm.time() - t0) / 60.:.2f} min')

        # Time array
        time = np.arange(Nframes[i]) * t_steps[i]
        # Total energy
        E = SNP_EK + SNP_EU
        # Plotting
        fig, axs = plt.subplots(1, 1)
        axs.plot(time[1:], np.abs((E[1:] - E[:-1]) / E[:-1]), ',k', label=typ)
        axs.legend(frameon=False)
        axs.set_xlabel(r't [Myr]')
        axs.set_ylabel(r'Energy error')
        axs.set_yscale('log')
        axs.set_title(f'Resolution: {t_resol[i]} steps/Myr')
        plt.show()
    return SNP_XV

