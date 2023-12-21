# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:01:37 2023

@author: emide
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Functions for Burgers' and Linear Advection Equations

L = 1                           # Spatial domain length.
n = 2**7 + 1                    # Spatial grid size.
x_all = np.linspace(0, L, n)    # all of the x's


def plot_u_data(Z, title, ax=None, sample_columns = [0, 10, 20, 40, 80, 160, 320, 640]):
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
        Z_column_all = add_boundary_data(Z[:,j])
        ax.plot(x_all, Z_column_all, color=next(color), label=fr"$q(x,t_{{{j}}})$")

    ax.set_xlim(x_all[0], x_all[-1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.legend(loc=(1.05, .05))
    ax.set_title(title)

def add_boundary_data(data):
    ''' Adds periodic boundary conditions back into the data '''
    
    if data.ndim == 1: # if data is a vector
        boundary = data[0]
        data_full = np.concatenate([data.T, [boundary]])
    else: # if data is a matrix
        boundary = data[0, :]
        data_full = np.concatenate([data[:, :], [boundary]])
 
    return data_full

def plot_area_under_curve(Z, title, ax = None):
    """Visualize area under the curve of data as it changes in time.
    
    This uses the full timesteps of data given, NOT sample columns"""
    
    if ax is None:
        _, ax = plt.subplots(1, 1)
    
        
    # Instantiate area
    area = [] 
    
    r, v = Z.shape
    
    # Plot a few snapshots in time over the spatial domain.
    for time in range(len(v)):
        # Adding boundary condition back to the data set
        Z_column_all = add_boundary_data(Z[:, time])

        # Finding area under the curve
        area.append(integrate.trapezoid(Z_column_all, x_all))
    
    ax.plot(area)
    ax.set_ylim((0 ,1))
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Area Under Curve')
    ax.set_title(title)

