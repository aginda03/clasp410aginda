#!/usr/bin/env python3

# Andrew Inda
# CLASP 410

'''
N-layer atmosphere model with experiment routines for Lab 1.

The script builds a matrix system to balance radiation between the surface and layers.  
It solves for fluxes, converts them to temperatures, and plots results.  
Experiments test how emissivity, number of layers, Venus conditions, and nuclear winter 
affect surface temp. 

NOTE: Indiviudully run %run Lab1.py, then call the experiment functions as needed.
NOTE: plot titles not included, descriptions in lab report.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# constants
sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W m^-2 K^-4]

def nlayeratm(nlayers, epsilon=1.0, albedo=0.33, s0=1350, debug=False):
    """
    Solve the N-layer atmosphere problem for radiative energy balance.

    Parameters:
        nlayers : int
            Number of atmospheric layers (N).
        epsilon : float, default=1.0
            Emissivity of atmosphere layersn (0 < ε ≤ 1).
        albedo : float, default=0.33
            Fraction of incoming solar radiation reflected.
        s0 : float, default=1350
            Solar constant (W m^-2).
        debug : bool, default=False
            If True, prints the A matrix and b vector for verification.

    Returns:
        temps : numpy.ndarray
            Array of temperatures (K) for the surface (0) and each atmosphere layer (n).
        fluxes : numpy.ndarray
            Array of fluxes (W m^-2) for each layer and the surface.
    """
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # populate A
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:
                A[i, j] = -1*(i > 0) - 1
            else:
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m

    # fix surface row
    A[0, 1:] /= epsilon

    # fill b
    b[0] = -s0/4 * (1-albedo)

    if debug:
        print("A:\n", A)
        print("b:\n", b)

    # solve system
    fluxes = np.linalg.solve(A, b)

    # convert to temperatures
    temps = (fluxes/(epsilon*sigma))**0.25
    temps[0] = (fluxes[0]/sigma)**0.25

    return temps, fluxes


def exp_emissivity():
    """
    Lab 1, Q3(a)

    Runs the N-layer model for a single-layer atmosphere (N=1)
    across emissivity values and plots surface
    temperature versus emissivity. Also includes the baseline
    no-atmosphere case (N=0) for comparison.

    How to run:
    exp_emissivity()
    """
    eps_vals = np.linspace(0.05, 1.0, 50)  # test emissivity values from 0.05 to 1
    Ts = []
    for eps in eps_vals:
        temps, _ = nlayeratm(1, epsilon=eps)  # run the 1-layer model at this ε
        Ts.append(temps[0])  # grab surface temp each time

    # run the no-atmosphere case to compare against
    bare_earth, _ = nlayeratm(0, epsilon=1)
    bare_temp = bare_earth[0]

    # plot both results to see how ε changes surface T
    plt.plot(eps_vals, Ts, label="1-layer atmosphere")
    plt.axhline(y=bare_temp, color="r", linestyle="--", label=f"No atmosphere (~{bare_temp:.1f} K)")
    plt.xlabel("Emissivity")
    plt.ylabel("Surface Temperature (K)")
    plt.legend()
    plt.show()


def exp_layers(epsilon=0.255, max_layers=10):
    """
    Lab 1, Q3(b))

    Runs the N-layer model with a fixed emissivity and
    increases the number of layers. Plots surface
    temperature versus number of layers.

    Parameters ():
        epsilon : float, default=0.255
            Atmospheric emissivity, same for all layers.
        max_layers : int, default=10, can change as needed.
            Maximum number of layers to test.

    Produces:
        A line plot of surface temperature (K) versus number of layers.

    How to run:
    exp_layers(epsilon=0.255, max_layers=10)
    """
    layers = np.arange(1, max_layers+1)
    Ts = []
    for n in layers:
        temps, _ = nlayeratm(n, epsilon=epsilon)
        Ts.append(temps[0])  # store surface temp for each layer count
    # plot how surface temp goes up with more layers
    plt.plot(layers, Ts, marker="o")
    plt.xlabel("Number of Layers")
    plt.ylabel("Surface Temperature (K)")
    plt.show()


def exp_venus():
    """
    Lab 1, Q4 (Venus)

    Uses the N-layer model with solar constant S0 = 2600 W/m^2
    and emissivity ε = 1. Increments the number of layers until
    the surface temperature reaches ~700 K.
    Prints the number of layers required for Venus to reach
    ~700 K surface temperature.

    How to run:
    exp_venus()
    """
    target = 700.0
    s0 = 2600
    n = 1
    # keep adding layers until Venus surface hits ~700 K
    while True:
        temps, _ = nlayeratm(n, epsilon=1.0, s0=s0)
        if temps[0] >= target:
            break
        n += 1
    print(f"Venus needs about {n} perfectly absorbing layers for ~{target} K surface temp.")


def exp_nuclearwinter():
    """
    Lab 1, Q5 (Nuclear Winter)

    Modifies the N-layer model so that all incoming solar flux
    is absorbed by the top layer instead of the surface.
    Runs with N=5, ε=0.5, and S0=1350 W/m^2, then plots
    temperature versus altitude (layer index).

    How to run:
    exp_nuclearwinter()
    """
    nlayers = 5
    epsilon = 0.5
    s0 = 1350

    # build A and b like before but change where solar energy goes
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:
                A[i, j] = -1*(i > 0) - 1
            else:
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    A[0, 1:] /= epsilon

    # instead of the surface, dump all solar flux into the top layer
    b[-1] = -s0/4 * (1-0.33)

    fluxes = np.linalg.solve(A, b)
    temps = (fluxes/(epsilon*sigma))**0.25
    temps[0] = (fluxes[0]/sigma)**0.25

    # plot temps vs layer index to see inverted profile
    plt.plot(np.arange(nlayers+1), temps, marker="o")
    plt.xlabel("Layer index (0 = surface)")
    plt.ylabel("Temperature (K)")
    plt.show()

    print(f"Surface temperature under nuclear winter: {temps[0]:.2f} K")
