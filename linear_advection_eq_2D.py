# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:48:25 2023

@author: emide
"""

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
import scipy
import opinf
from opinf_finite_dif import ddt
import imageio

# Linear Advection Equation
# du(x,t) / dt = c * du(x,t) / dx

## Making the Full Order Model (FOM)

# Construct the spatial x domain.
Lx = 1                          # Spatial domain length.
nx = 200 # L*2**8 + 1           # Spatial grid size.
x_all = np.linspace(0, Lx, nx+1)  # all of the x's
x = x_all[0:-1]                 # the unknown u's (u_n is equal to u_0)
dx = x[1] - x[0]                # Spatial resolution.
advection_u = -1 # m/s

# Construct the spatial y domain.
Ly = 1                          # Spatial domain length.
ny = 150 # L*2**8 + 1           # Spatial grid size.
y_all = np.linspace(0, Ly, ny+1)  # all of the x's
y = y_all[0:-1]                 # the unknown u's (u_n is equal to u_0)
dy = y[1] - y[0]                # Spatial resolution.
advection_v = -1 # m/s


# Construct the temporal domain.
t0 = 0                          # Initial simulation time
tf = 2                          # Temporal domain length (final simulation time).
dt = 0.001                      # Temporal resolution
t = np.arange(0, tf, dt)        # Temporal grid.
nt = len(t)

print(f"Spatial step size δx = {dx}")
print(f"Temporal step size δt = {dt}")

# Construct the full-order state matrix A for x domain.
factor_x = int(advection_u / 2 / dx)
diags_x = np.array([-1, 0, 1]) * factor_x
A_x = sparse.diags(diags_x, [-1, 0, 1], (nx, nx), dtype = int).toarray()
A_x[0, -1] = -1 * factor_x
A_x[-1, 0] = 1 * factor_x

# Construct the full-order state matrix A for y domain.
factor_y = int(advection_v / 2 / dy)
diags_y = np.array([-1, 0, 1]) * factor_y
A_y = sparse.diags(diags_y, [-1, 0, 1], (ny, ny), dtype = int).toarray()
A_y[0, -1] = -1 * factor_y
A_y[-1, 0] = 1 * factor_y


# Gaussian function + parameters
# This will be the initial condition of u

T0 = np.zeros((nx, ny))
for i in range(len(x)):
    for j in range(len(y)):
        # T0[i, j] = np.exp((x[i]-.5)**2/2)*np.exp((y[j]-.5)**2/2)
        T0[i, j] = -(x[i]-Lx/2)**2*2 + -(y[j]-Ly/2)**2*2 + 1

# Plot initial condition
fig = plt.pcolormesh(x, y, T0.T)
plt.title('Initial Condition $T_{0}$ at t = 0')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.axis('scaled')

# %%

def func_2D(T_new, T_old, Ax, Ay):
    return T_old - T_new + dt/2*Ax @ (T_old + T_new) + dt/2*(Ay @ (T_old + T_new).T).T

def implicit_midpoint(t, u0, A_x, A_y):
    """Solve the system

        du / dt = Au(t),    u(0) = u0,

    over a uniform time domain via the implicit midpoint method.

    Parameters
    ----------
    t : (k,) ndarray
        Uniform time array over which to solve the ODE.
    u0 : (n,) ndarray
        Initial condition.
    A : (n, n) ndarray
        State matrix.

    Returns
    -------
    u : (n, k) ndarray
        Solution to the ODE at time t; that is, u[:,j] is the
        computed solution corresponding to time t[j].
    """
    # Check and store dimensions.
    k = len(t)
    nx = len(x)
    ny = len(y)
    
    # Check that the time step is uniform.
    dt = t[1] - t[0]
    assert np.allclose(np.diff(t), dt)

    # Instantiating array
    u = np.empty((nx, ny, k))
    u[:, :, 0] = u0
    
    # Solve the problem by stepping in time.
    for j in range(1, k):
        u[:, :, j] = scipy.optimize.newton_krylov(lambda y:func_2D(y, u[:, :, j-1], A_x, A_y), u[:, :, j-1], verbose = True)

    return u

# Solving for FOM data using implicit midpoint method
T = implicit_midpoint(t, T0, A_x, A_y)

def add_boundary_data(data):
    ''' Adds periodic boundary conditions back into the data '''
    
    u, v, w = data.shape
    data_full = np.zeros((u+1, v+1, w))
    data_full[:u, :v, :] = data
    
    for time in range(w):
        boundary_x = data_full[0, :, time]
        data_full[-1, :, time] = boundary_x
        boundary_y = data_full[:, 0, time]
        data_full[:, -1, time] = boundary_y

    return data_full

T_full = add_boundary_data(T)

# %%
# Sample time steps for u plots and data

def plot_u_data(Z, sample_columns = np.linspace(0, tf/dt-10, 21).astype('int'), title = 'FOM'):
    """Visualize displacement data in space and time
    Input: 
        purely solved data (Z) without the boundary added in the last entry """

    i = 1
    frames = []        
    # Plot a few snapshots in time over the spatial domain.
    for j in sample_columns:
        fig, ax = plt.subplots(1, 1)
        # Adding back in boundary conditions
        step_to_time = np.round(j*dt, 2)
        Z_all = add_boundary_data(Z)
        c = ax.pcolormesh(x_all, y_all, Z_all[:, :, j].T)
    
        ax.set_title(fr"$T(x, y, t_{{{step_to_time}}})$")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(c)
        ax.axis('scaled')
        
        fname = './img/' + title + f'_{i}.png'
        
        plt.savefig(fname)
        
        i += 1
        
        image = imageio.v2.imread(fname)
        frames.append(image)
        plt.close()
    
    imageio.mimsave('./' + title + '.gif', # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second
    
plot_u_data(T)

# %%


def plot_area_under_curve(Z, title, ax = None):
    """Visualize area under the curve of data as it changes in time.
    
    This uses the full timesteps of data given, NOT sample columns"""
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
    
    a, b, c = Z.shape    # a = x grid, b = y grid, c = time grid
    
    # Adding boundary condition back to the data set
    Z_all = add_boundary_data(Z)
    
    # Instantiate area
    area_in_time = []
    
    # Plot a few snapshots in time over the spatial domain.
    for time in range(c):
        running_area = 0
        
        for y_col in range(b):
            # finding area in each x column in each time then adding it to 
            # running area total
            running_area += integrate.trapezoid(Z_all[:, y_col, time], x_all)
            
        
        print(running_area)
        # Once have sum, put into area
        area_in_time.append(running_area)
    
    print(area_in_time)
    ax.plot(area_in_time)
    ax.set_ylim((0, 150))
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Area Under Curve')
    ax.set_title(title)

## Plotting FOM and wave area over time
plot_area_under_curve(T, 'Area Under Curve of FOM')

# %% 
# Now putting everything into vector form to prep for ROM/minimization

T_timevec = np.zeros([nx*ny, nt])

for time in range(nt):
    T_timevec[:, time] = T[:, :, time].flatten()
    
B = np.zeros([nx*ny, nx*ny], dtype = int)


#%%

for ii in range(ny):
    B[ii*nx:(ii+1)*nx, ii*nx:(ii+1)*nx] = A_x
    
    
# %% 

C = np.zeros([nx*ny, nx*ny], dtype = int)

print(C.shape)


# factor_y = advection_v / 2 / dy
# diags_y = np.array([-1, 0, 1]) * factor_y
# A_y = sparse.diags(diags_y, [-1, 0, 1], (ny, ny)).toarray()
# A_y[0, -1] = -1 * factor_y
# A_y[-1, 0] = 1 * factor_y


k = nx*2 + 2
full_dy_vec = np.zeros([1, k], dtype = int)


print(full_dy_vec)

full_dy_vec[0, 0] = -1 * factor_y
full_dy_vec[0, -1] = 1 * factor_y
print(full_dy_vec.shape)
print(full_dy_vec)

# %%

# C_can = C_canonical AKA the matrix that will be propogated to 
C_can = np.zeros((ny, nx*ny), dtype = int)
print(C_can.shape)

# boundary cells
C_can[0, :201] = full_dy_vec[0, 201:]
C_can[-1, -201:] = full_dy_vec[0, :201]

# inner cells
for jj in range(1, ny-1):
    C_can[jj, (jj-1)*nx:(jj-1)*nx + k] = full_dy_vec
    print(jj)
    
# now propogate this into the actual C array
# for ii in range(1, nx - 1):
C[:150, :] = C_can[:, :]            # the first and actual C_can in C  

# next canonicals are offset in each column by nx*ii
for ii in range(1, nx - 1):
    C[ii*ny:(ii+1)*ny, ii:] = C_can[:, :-ii]
    C[ii*ny:(ii+1)*ny, :ii] = C_can[:, -ii:]

#%%    

test = B + C

dT_ = test @ T_timevec

# %% 
# OpInf Reduced Order Modeling (ROM)

# Only want to train the ROM with the first 500 snapshots
k = 1800
t_train = t[:k]
T_timevec_train = T[:, :k]

# Compute the POD basis, using the residual energy decay to select r
basis = opinf.pre.PODBasis().fit(T_timevec, residual_energy=1e-6)
print(basis)

# %%

# Check the decay of the singular values
basis.plot_svdval_decay()
plt.xlim(0, 100)

# Check the decay of the residual energy based on the singular values
basis.plot_residual_energy(threshold=1e-6)
plt.xlim(0, 100)
plt.show()

# Train the model
rom = opinf.ContinuousOpInfROM(modelform="A")

rom.fit(basis=basis, states=T_timevec, ddts = dT_, regularizer = 0)
print(rom)

# Express the initial condition in the coordinates of the basis

# first need to flatten T0 into vector format
T0_vec = T0.flatten()

T0_ = basis.encode(T0_vec)

def implicit_midpoint_oneA(t, u0, A):
    """Solve the system

        du / dt = Au(t),    u(0) = u0,

    over a uniform time domain via the implicit midpoint method.

    Parameters
    ----------
    t : (k,) ndarray
        Uniform time array over which to solve the ODE.
    u0 : (n,) ndarray
        Initial condition.
    A : (n, n) ndarray
        State matrix.

    Returns
    -------
    u : (n, k) ndarray
        Solution to the ODE at time t; that is, u[:,j] is the
        computed solution corresponding to time t[j].
    """
    # Check and store dimensions.
    k = len(t)
    n = len(u0)
    assert A.shape == (n, n)

    I = np.eye(n) # Identity Matrix

    # Check that the time step is uniform.
    dt = t[1] - t[0]
    assert np.allclose(np.diff(t), dt)

    # Instantiating array
    u = np.empty((n, k))
    u[:,0] = u0.copy()
    
    # Solve the problem by stepping in time.
    for j in range(1, k):
        # u[:, j], __ = scipy.sparse.linalg.lgmres(I - dt/2*A, (I + dt/2*A) @ u[:, j-1], atol = 1e-12)
        u[:, j] = scipy.linalg.solve(I - dt/2*A, (I + dt/2*A) @ u[:, j-1])

    return u

# Solve the reduced-order model using Implicit Midpoint method
T_ROM_vec = basis.decode(implicit_midpoint_oneA(t, T0_, rom.A_.entries))


T_ROM = np.reshape(T_ROM_vec, (nx, ny, nt))

plot_u_data(T_ROM, title = 'ROM')

# %% 
# Solving for the minimum matrix A that is Skew Symmetric

import cvxpy as cp

# Constructing matrix variable A
r = basis.r
A_min = cp.Variable((r, r))

# Least squares regularization term
gamma = 100

# Putting full dimensional ROM data into its reduced dimension

T_reduced = basis.encode(T_timevec)
Tdot_reduced = basis.encode(dT_)

# Creating the objective (to be minimized)
objective = cp.sum_squares(Tdot_reduced - A_min @ T_reduced) + gamma*cp.sum_squares(A_min)

# Constraints on objective
constraints = [A_min.T == -A_min, cp.diag(A_min) == 0]

# Instantiating cvxpy problem and solving
prob = cp.Problem(cp.Minimize(objective), constraints)
result = prob.solve(solver = 'SCS', verbose = True) #solver = 'SCS', eps = 1e-12

A_skew = A_min.value

# solving for data with new A_skew matrix with the implicit midpoint
T_skew_vec = basis.decode(implicit_midpoint_oneA(t, T0_, A_skew))
T_skew = np.reshape(T_skew_vec, (nx, ny, nt))

# %%
# plotting FOM, ROM, and Skew next to each other
plot_u_data(T_skew, title = 'Skew')
plt.tight_layout() 
plt.show()
 
print('A_skew eigenvalues:', np.linalg.eigvals(A_skew))

# %%
# Plot Area Under the Curve for FOM, ROM, and Skew Datasets
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (8, 4))

plot_area_under_curve(u_FOM, 'FOM', ax1)
plot_area_under_curve(u_ROM, 'ROM', ax2)
plot_area_under_curve(u_skew, 'Skew Area', ax3)
plt.tight_layout() 

# %%
# Plot Eigenvalues of A_ROM and A_skew 

eig_A_skew = np.linalg.eigvals(A_skew)
eig_A_ROM = np.linalg.eigvals(rom.A_.entries)

fig, ax = plt.subplots(1, 1)
ax.scatter(eig_A_skew.real, eig_A_skew.imag, color = 'blue', label = 'Optimized Skew Symmetric')
ax.scatter(eig_A_ROM.real, eig_A_ROM.imag, color = 'red', label = 'Operator Inference ROM')
ax.legend()
ax.axvline(x = 0, ls = '--', c = 'grey')
ax.axhline(y = 0, ls = '--', c = 'grey')
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_title('Eigenvalues of Matrix A')

# %% 
# Checking if Skew Matrix is Within Tolerance

A_skew_round = np.round(A_skew, 7)
hmm = A_skew_round == -A_skew_round.T
np.isclose(A_skew, -A_skew.T, rtol=1e-8, atol=1e-08, equal_nan=False)

# %% Predicting Further in Time for both Skew and ROM

tf_predict = 5                              # Temporal domain length (final simulation time).
t_predict = np.arange(0, tf_predict, dt)    # Temporal prediction grid.

u_ROM_prediction = basis.decode(implicit_midpoint(t_predict, u0_, rom.A_.entries))

u_skew_prediction = basis.decode(implicit_midpoint(t_predict, u0_, A_skew))

# Solving for FOM data using implicit midpoint method
u_FOM_prediction = implicit_midpoint(t_predict, u0, A)

# %%

sam_col = np.linspace(0, tf_predict*1000-10, 10).astype('int')

fig = plt.figure(figsize = (6, 10))
ax1, ax2, ax3 = fig.subplots(3, 1)
plot_u_data(u_FOM_prediction, "FOM Prediction", ax1, sam_col)
plot_u_data(u_ROM_prediction, "ROM Prediction", ax2, sam_col)
plot_u_data(u_skew_prediction, "Skew Prediction", ax3, sam_col)
plt.tight_layout() 


fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize = (4, 10))
plot_area_under_curve(u_FOM_prediction, "FOM Prediction", ax1)
plot_area_under_curve(u_ROM_prediction, 'ROM Prediction', ax2)
plot_area_under_curve(u_skew_prediction, 'Skew Prediction', ax3)
plt.tight_layout() 


