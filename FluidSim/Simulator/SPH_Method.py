import numpy as np
import matplotlib as mpl

pi = np.pi

def kernal(x, y, z, smooth): # W(r;h) function
    # x      = rx = Matrix of all x positions
    # y      = ry = Matrix of all y positions
    # z      = rz = Matrix of all z positions
    # smooth = h  = Smoothing lenght
    # w      = w  = Evaluated function
    
    r = np.sqrt(x**2 + y**2 + z**2) # Calculate the value of ||r||
    
    w = (1.0 / ( smooth * np.sqrt( pi )))**3 * np.exp( -r**2 / smooth**2) # W function
    
    return w

def gradientKernal(x, y, z, smooth): # \7 W(r;h) function
    # x        = rx = Matrix of all x positions
    # y        = ry = Matrix of all y positions
    # z        = rz = Matrix of all z positions
    # smooth   = h  = Smoothing lenght
    # wx wy wz = w  = Evaluated function (gradient)
    
    r = np.sqrt(x**2 + y**2 + z**2) # Calculate the value of ||r||
    
    n = -2 * np.exp( -r**2 / smooth**2) / smooth**5 / pi**(3 / 2)  # \7 W function
    
    wx = n * x
    wy = n * y
    wz = n * z
    
    return wx, wy, wz

def pairwiseSeparation(ri, rj): # Separation of two sets of coordinates
    # ri         = M by 3 matrix where 3 is the xyz positions and M the amount of particles
    # rj         = N by 3 martix where 3 is the xyz positions and N the amount of particles
    # dx, dy, dz = M by N matrix of position separations
    
    M = ri.shape[0] # M = lenght of ri (particles)
    N = rj.shape[0] # N = lenght of rj (particles)
    
    # positions of each ri x,y,z in a 1D matrix, lenght of M (particles)
    rix = ri[:,0].reshape((M,1)) # first value (x) reshape => 1D matrix
    riy = ri[:,1].reshape((M,1)) # first value (y) reshape => 1D matrix
    riz = ri[:,2].reshape((M,1)) # first value (z) reshape => 1D matrix
    
    # positions of each rj x,y,z in a 1D matrix, lenght of N (particles)
    rjx = rj[:,0].reshape((N,1)) # first value (x) reshape => 1D matrix
    rjy = rj[:,1].reshape((N,1)) # first value (y) reshape => 1D matrix
    rjz = rj[:,2].reshape((N,1)) # first value (z) reshape => 1D matrix
    
    # matrices that store the separations ri - rj
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    
    return dx, dy, dz

def density(r, position, mass, smooth): # Calculate density at a certain position in space
    # r        = r   = M by 3 matrix of sampling positions
    # position = pos = N by 3 martix of SPH paticle positions
    # mass     = m   = Mass of a particle
    # smooth   = h   = Smoothing lenght
    # denst    = p   = 1D matrix of acceleations
    
    M = r.shape[0] # Lenght of r (positions)
    
    dx, dy, dz = pairwiseSeparation(r, position); # ri - rj using the custom function
    
    denst = np.sum( mass * kernal(dx, dy, dz, smooth), 1 ).reshape((M,1)) # Sum(Sigma) mass and W(ri - j;h)
    
    return denst

def pressure(denst, constant, index):
    # denst    = p = Vector of densities
    # constant = k = Equation of state(polytropic) constant
    # index    = n = Polytropic index
    # pressure = P = Pressure
    
    pressure = constant * denst**(1+1/index)
    
    return pressure

def acceleration(position, velocity, mass, smooth, constant, index, lmdba, visc): # Calculate the acceleration
    # position = pos = M by 3 matrix of positions
    # velocity = vel = N by 3 martix of velocities
    # mass     = m   = Mass of a particle
    # smooth   = h   = Smoothing lenght
    # constant = k   = Equation of state(polytropic) constant
    # index    = n   = Polytropic index
    # lmdba    = f   = External force constant
    # visc     = nu  = Viscosity of the fluid
    # accel    = a   = N by 3 matrix of accelerations
    
    N = position.shape[0] # Lenght of r (positions)
    
    denst = density(position, position, mass, smooth) # Get the density
    
    press = pressure(denst, constant, index) # Get the pressure
    
    dx, dy, dz = pairwiseSeparation(position, position); # ri - rj using the custom function
    dWx, dWy, dWz = gradientKernal(dx, dy, dz, smooth); # gradient using \7 W(ri-rj;h)   
    
     # Add Pressure contribution to accelerations (the part in backets)
    accelX = - np.sum( mass * ( press / denst**2 + press.T / denst.T**2  ) * dWx, 1).reshape((N,1))
    accelY = - np.sum( mass * ( press / denst**2 + press.T / denst.T**2  ) * dWy, 1).reshape((N,1))
    accelZ = - np.sum( mass * ( press / denst**2 + press.T / denst.T**2  ) * dWz, 1).reshape((N,1))
    
    accel = np.hstack((accelX, accelY, accelZ)) # Pack together the acceleration components
    
    accel += -lmdba * position - visc * velocity # Add external potential force and viscosity using 
                                                 # Euler equation for an ideal fluid
        
    return accel

