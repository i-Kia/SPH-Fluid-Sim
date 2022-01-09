import numpy as np
import matplotlib as mpl
import SPH_Method as sph

pi = np.pi

def setStartingConditions():
    default = [42, 400, 12, 0.04, 2, 0.1, 0.1, 1, 1]
    settingsList = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    inputQuestion = [
        '[default == 42  ] Random seed         : ',
        '[default == 400 ] Amount of particles : ',
        '[default == 12  ] Simulation steps    : ',
        '[default == 0.04] Timestep            : ',
        '[default == 2   ] Overall mass        : ',
        '[default == 0.1 ] Smoothing lenght    : ',
        '[default == 0.1 ] Polytropic constant : ',
        '[default == 1   ] Polytropic index    : ',
        '[default == 1   ] Viscosity           : ']
    for setting in range(len(settingsList)):
        Input = input(inputQuestion[setting])
        if Input == '':
            settingsList[setting] = default[setting]
        else:
            settingsList[setting] = int(Input)
    return settingsList

def main():
    np.random.seed(42)            # set the random number generator seed

    StartCond = setStartingConditions()

    np.random.seed(StartCond[0])  # set the random number generator seed
    ParticleNum = StartCond[1]    # Number of particles
    time        = 0               # current time of the simulation
    tEnd        = StartCond[2]    # time at which simulation ends
    dtime       = StartCond[3]    # timestep
    Mass        = StartCond[4]    # overall fluid mass
    smooth      = StartCond[5]    # smoothing length
    constant    = StartCond[6]    # equation of state constant
    index       = StartCond[7]    # polytropic index
    visc        = StartCond[8]    # damping

    #######
    print(StartCond)
    #######
    
    #lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
    lmdba = 0

    mass = Mass/ParticleNum # single particle mass
    position = np.random.randn(ParticleNum, 3) # randomly selected positions and velocities
    velocity = np.zeros(position.shape) # Set up velocyties as zero
    
    accel = sph.acceleration(position, velocity, mass, smooth, constant, index, lmdba, visc )
    
    timeSteps = int(np.ceil(tEnd/dtime))
    
    # Main Loop
    for t in range(timeSteps):
        # (1/2) kick
        velocity += accel * dtime/2
    
        # drift
        position += velocity * dtime
    
        # update accelerations
        accel = sph.acceleration(position, velocity, mass, smooth, constant, index, lmdba, visc )
    
        # (1/2) kick
        velocity += accel * dtime/2
    
        # update time
        time += dtime