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


# Linear Advection Equation
# du(x,t) / dt = c * du(x,t) / dx

## Making the Full Order Model (FOM)

# Construct the spatial domain.
L = 1                           # Spatial domain length.
n = 200 # L*2**8 + 1            # Spatial grid size.
x_all = np.linspace(0, L, n)    # all of the x's
x = x_all[0:-1]                 # the unknown u's (u_n is equal to u_0)
dx = x[1] - x[0]                # Spatial resolution.
advection_v = -1 # m/s

# Construct the temporal domain.
t0 = 0                          # initial simulation time
tf = 2                          # Temporal domain length (final simulation time).
dt = 0.001                      # Temporal resolution
t = np.arange(0, tf, dt)        # Temporal grid.

print(f"Spatial step size δx = {dx}")
print(f"Temporal step size δt = {dt}")

# Construct the full-order state matrix A.
factor = advection_v / 2 / dx
diags = np.array([-1, 0, 1]) * factor
A = sparse.diags(diags, [-1, 0, 1], (n-1, n-1)).toarray()
A[0, -1] = -1 * factor
A[-1, 0] = 1 * factor

# Gaussian function + parameters
# This will be the initial condition of u
a = 2
b = 0.5
c = 0.1
u0 = a*np.exp(-(x-b)**2/(2*c**2))

# Plot initial condition
plt.plot(x, u0)
plt.title('Initial Condition $U_{0}$ at t = 0')
plt.xlabel('x')
plt.ylabel('u(x, t = 0)')

def implicit_midpoint(t, u0, A):
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

# Solving for FOM data using implicit midpoint method
u_FOM = implicit_midpoint(t, u0, A)

def add_boundary_data(data):
    ''' Adds periodic boundary conditions back into the data '''
    
    if data.ndim == 1: # if data is a vector
        boundary = data[0]
        data_full = np.concatenate([data.T, [boundary]])
    else: # if data is a matrix
        boundary = data[0, :]
        data_full = np.concatenate([data[:, :], [boundary]])
 
    return data_full

# Sample time steps for u plots and data

def plot_u_data(Z, title, ax = None, sample_columns = np.linspace(0, tf*1000-10, 10).astype('int')):
    """Visualize displacement data in space and time
    Input: 
        purely solved data (Z) without the boundary added in the last entry """
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
        
    # Adding color to timesteps
    color = iter(plt.cm.inferno_r(np.linspace(.05, 1, len(sample_columns))))
    
    # Plot a few snapshots in time over the spatial domain.
    for j in sample_columns:
        # # Adding back in boundary conditions
        step_to_time = np.round(j*dt, 2)
        Z_column_all = add_boundary_data(Z[:,j])
        ax.plot(x_all, Z_column_all, color=next(color), label=fr"$q(x,t_{{{step_to_time}}})$")

    ax.set_xlim(x_all[0], x_all[-1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.legend(loc=(1.05, .05))
    ax.set_title(title)


def plot_area_under_curve(Z, title, ax = None):
    """Visualize area under the curve of data as it changes in time.
    
    This uses the full timesteps of data given, NOT sample columns"""
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
    
    r, s = Z.shape    # r = x grid, s = t grid
    # Instantiate area
    area = [] 
    
    # Plot a few snapshots in time over the spatial domain.
    for time in range(s):
        # Adding boundary condition back to the data set
        Z_column_all = add_boundary_data(Z[:, time])

        # Finding area under the curve
        area.append(integrate.trapezoid(Z_column_all, x_all))
    
    ax.plot(area)
    ax.set_ylim((0 ,1))
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Area Under Curve')
    ax.set_title(title)

## Plotting FOM and wave area over time
__, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 10))

plot_u_data(u_FOM, 'Full Order Model (FOM)', ax1)
plot_area_under_curve(u_FOM, 'Area Under Curve of FOM', ax2)

# %% 
# OpInf Reduced Order Modeling (ROM)

# Only want to train the ROM with the first 500 snapshots
k = 1600
t_train = t[:k]
U = u_FOM[:, :k]

# Need time derivative du/dt
Udot_train = (U[:, 1:] - U[:, :-1])/dt
U_train = u_FOM[:, 1:k] # U[:, 1:] 

# dt = t[1] - t[0]
# Udot_test = ddt(U, dt, order = 6)
# Udot_test = Udot_test[:, :-1]

# Udot_train = Udot_test

print(Udot_train.shape)
print(U_train.shape)

# Compute the POD basis, using the residual energy decay to select r
basis = opinf.pre.PODBasis().fit(U, residual_energy=1e-6)
print(basis)

# Check the decay of the singular values
basis.plot_svdval_decay()
plt.xlim(0, 25)

# Check the decay of the residual energy based on the singular values
basis.plot_residual_energy(threshold=1e-6)
plt.xlim(0, 25)
plt.show()

# Train the model
rom = opinf.ContinuousOpInfROM(modelform="A")
rom.fit(basis=basis, states=U_train, ddts=Udot_train, regularizer = 0)
print(rom)

# Express the initial condition in the coordinates of the basis
u0_ = basis.encode(u0)

# Solve the reduced-order model using Implicit Midpoint method
u_ROM = basis.decode(implicit_midpoint(t, u0_, rom.A_.entries))

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 10))
plot_u_data(u_FOM, "Full-order model solution", ax1)
plot_u_data(u_ROM, "Reduced-order model solution", ax2)
plt.tight_layout()
plt.show()

# %% 
# Solving for the minimum matrix A that is Skew Symmetric

import cvxpy as cp

# Constructing matrix variable A
r = basis.r
A_min = cp.Variable((r, r))

# Least squares regularization term
gamma = 0

# Putting full dimensional FOM data into its reduced dimension
u_FOM_dot = ddt(u_FOM, dt, order = 6)

u_reduced = basis.encode(u_FOM)
udot_reduced = basis.encode(u_FOM_dot)

# Creating the objective (to be minimized)
objective = cp.sum_squares(udot_reduced - A_min @ u_reduced) + gamma*cp.sum_squares(A_min)

# Constraints on objective
constraints = [A_min.T == -A_min, cp.diag(A_min) == 0]

# Instantiating cvxpy problem and solving
prob = cp.Problem(cp.Minimize(objective), constraints)
result = prob.solve(solver = 'OSQP', verbose = True) #solver = 'SCS', eps = 1e-12

A_skew = A_min.value

# solving for data with new A_skew matrix with the implicit midpoint
u_skew = basis.decode(implicit_midpoint(t, u0_, A_skew))

# plotting FOM, ROM, and Skew next to each other
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(6, 10))
plot_u_data(u_FOM, "Full-order Model Solution", ax1)
plot_u_data(u_ROM, "Reduced-order Model Solution", ax2)
plot_u_data(u_skew, "Optimized Reduced Skew Model Solution", ax3)
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


