
import numpy as np
import matplotlib.pyplot as plt

#Define the domain 
#Domain Size
Lx, Ly = 2.0, 2.0

#Number of grid points
Nx, Ny = 100, 100

#Create grid 
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

#Create mesh
X, Y = np.meshgrid(x,y)
#print(X.shape)

#Initial condition: Gaussian blob
x0, y0 = 0.5, 1.0
sigma = 0.1
u = np.exp(-((X-x0)**2 + (Y - y0)**2)/ sigma**2)

#visualization of initial condition (Figure 1)
plt.contour(X, Y, u)
plt.colorbar()
plt.title("Initial Condition")

#Parameters
#Velocity (constant flow to the right) 
a = 1.0 #x direction
b = 0.0 #y direction

#Diffusion coefficient
D = 0.05
#Time stepping 
dt = 0.002

#Derivatives approximation
dx = x[1] - x[0]
dy = y[1] - y[0]

#One time step function
def step(u):
    u_new = u.copy()

    #Interior points only
    for i in range(1, Nx- 1):
        for j in range(1, Ny-1):
            #First derivatives (upwind for stability)
            du_dx = (u[i, j] - u[i-1, j]) / dx
            du_dy = (u[i, j] - u[i, j-1]) / dy
            #Second derivatives (central difference)
            d2u_dx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            d2u_dy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            #Update equation
            #- dt*(a * du_dx + b * du_dy) for transport and diffusion dt*D*(d2u_dx2 + d2u_dy2)
            u_new[i, j] = u[i, j]\
                - dt*(a * du_dx + b * du_dy) \
                + dt*D*(d2u_dx2 + d2u_dy2)
    return u_new        

#Time steps
Nt = 1000


for n in range(Nt):
    u = step(u)

#Final plot (figure 2) 
plt.contour(X, Y, u)
plt.colorbar()
plt.title("Initial Vs Final State")
plt.show()

#create circular obstacle (fake airfoil)
xc, yc = 1.0, 1.0
r = 0.2

mask = (X - xc)**2 + (Y-yc)**2 < r**2

def step(u):
    u_new = u.copy()
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            
            # 👇 THIS must be inside BOTH loops
            if mask[i, j]:
                u_new[i, j] = 0
                continue
            
            # First derivatives
            du_dx = (u[i, j] - u[i-1, j]) / dx
            du_dy = (u[i, j] - u[i, j-1]) / dy
            
            # Second derivatives
            d2u_dx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            d2u_dy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            
            # Update
            u_new[i, j] = u[i, j] \
                - dt*(a * du_dx + b * du_dy) \
                + dt*D*(d2u_dx2 + d2u_dy2)
    
    return u_new


#to alter the motion of the fluid, que can increase the simulation time Nt > 200, increase its velocity >1.0 or increade dt slightly 0.005
