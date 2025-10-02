#!/usr/bin/env python3
'''
Lab 2: Population Control
Andrew Inda

This script models Lab 2, solves Lotka–Volterra equations
for competition and predator–prey systems. Both Euler and
RK8 solvers are used.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use("seaborn-v0_8")


## Derivative functions


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    """
    Competition model ODEs. Two species competing.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species, assuming N is a two element vector
    N1, N2 = N
    # Logistic growth - competition terms
    dN1dt = a * N1 * (1 - N1) - b * N1 * N2
    dN2dt = c * N2 * (1 - N2) - d * N1 * N2
    return [dN1dt, dN2dt]


def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    """
    Predator–prey model ODEs. Prey growth and predator hunting them.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Prey grows, eaten by predator
    dN1dt = a * N1 - b * N1 * N2
    # Predator dies, grows when eating prey
    dN2dt = -c * N2 + d * N1 * N2
    return [dN1dt, dN2dt]


## Euler solver


def euler_solve(func, N1_init=0.3, N2_init=0.6, dT=0.1, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    Euler solver (fixed step). Models the populations step by step.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Build time array
    time = np.arange(0, t_final + dT, dT)
    # Storage arrays
    N1 = np.zeros_like(time)
    N2 = np.zeros_like(time)
    # Set initial values
    N1[0], N2[0] = N1_init, N2_init

    # March forward in time
    for i in range(1, len(time)):
        dN1dt, dN2dt = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        # Euler update: new = old + step * slope
        N1[i] = N1[i-1] + dT * dN1dt
        N2[i] = N2[i-1] + dT * dN2dt
    return time, N1, N2


## RK8 solver


def solve_rk8(func, N1_init=0.3, N2_init=0.6, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    RK8 solver (DOP853, adaptive step). Models smoother populations and takes smaller steps when necessary.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : max step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Call SciPy ODE solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=(a, b, c, d), method="DOP853", max_step=dT)
    return result.t, result.y[0], result.y[1]


## LAB QUESTIONS 

'''
for all of these functions, run with 

%run Lab2.py
function_name()
'''

## Question 1


def run_Q1_comp():
    """
    Q1a: Compares the Euler vs RK8 for competition model.

    Inputs: none (params inside function)
    Outputs: plot of N1 and N2 for both solvers
    """
    # Params and initial conditions
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.3, 0.6

    # Run Euler
    t_euler, N1_euler, N2_euler = euler_solve(
        dNdt_comp, N1_init, N2_init, dT=0.1, t_final=100, a=a, b=b, c=c, d=d)

    # Run RK8
    t_rk8, N1_rk8, N2_rk8 = solve_rk8(
        dNdt_comp, N1_init, N2_init, dT=10, t_final=100, a=a, b=b, c=c, d=d)

    # Plot both solvers
    plt.figure(figsize=(8,5))
    plt.plot(t_euler, N1_euler, "r--", label="N1 Euler")
    plt.plot(t_euler, N2_euler, "b--", label="N2 Euler")
    plt.plot(t_rk8, N1_rk8, "r", label="N1 RK8")
    plt.plot(t_rk8, N2_rk8, "b", label="N2 RK8")
    plt.xlabel("Time (years)")
    plt.ylabel("Population (normalized)")
    plt.legend()
    plt.show()


def run_Q1_predprey():
    """
    Q1b: Compares the Euler vs RK8 for predprey model.

     Inputs: none (params inside function)
     Outputs: plot of N1 and N2 for both solvers
    """
    # Parameters given in the lab assignment
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.3, 0.6
    t_final = 100

    # Euler solver (dt = 0.05 yr)
    t_euler, N1_euler, N2_euler = euler_solve(
        dNdt_predprey, N1_init, N2_init,
        dT=0.05, t_final=t_final,
        a=a, b=b, c=c, d=d
    )

    # RK8 solver (adaptive, max step = 5 yr just for output density)
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_predprey, N1_init, N2_init,
        dT=5.0, t_final=t_final,
        a=a, b=b, c=c, d=d
    )

    # Plot results
    plt.figure(figsize=(9,6))
    plt.plot(t_euler, N1_euler, 'g--', label="Prey N1 (Euler)")
    plt.plot(t_euler, N2_euler, 'm--', label="Predator N2 (Euler)")
    plt.plot(t_rk, N1_rk, 'g', label="Prey N1 (RK8)")
    plt.plot(t_rk, N2_rk, 'm', label="Predator N2 (RK8)")
    
    plt.xlabel("Time (years)")
    plt.ylabel("Population (normalized)")
    plt.title("Q1 Predator–Prey: Euler vs RK8")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()



## Question 2


def run_Q2(t_final=10):
    """
    Q2: Run 4 cases of competition model with RK8.

    Inputs:
        t_final : total time (years)
    Outputs:
        2x2 subplot figure with N1 (dashed) and N2 (solid)
    """
    # Define 4 cases with different starts and coefficients, chose these arbitrarily to show different behaviors
    cases = [
        {"N1_init": 0.9, "N2_init": 0.1, "a": 1, "b": 2, "c": 1, "d": 3, "label": "Case A"},
        {"N1_init": 0.1, "N2_init": 0.9, "a": 1, "b": 2, "c": 1, "d": 3, "label": "Case B"},
        {"N1_init": 0.5, "N2_init": 0.5, "a": 1, "b": 5, "c": 1, "d": 1, "label": "Case C"},
        {"N1_init": 0.5, "N2_init": 0.5, "a": 1, "b": 0.5, "c": 1, "d": 0.5, "label": "Case D"},
    ]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Loop through each case
    for i, cs in enumerate(cases):
        col = colors[i % len(colors)]
        ax = axes[i]

        # Run RK8
        t, N1, N2 = solve_rk8(
            dNdt_comp, N1_init=cs["N1_init"], N2_init=cs["N2_init"],
            t_final=t_final, dT=5,
            a=cs["a"], b=cs["b"], c=cs["c"], d=cs["d"])
        
        # Plot results for this case
        ax.plot(t, N1, '--', color=col, label='N1')
        ax.plot(t, N2, '-',  color=col, label='N2')

        ax.set_title(cs["label"])
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Population")
        ax.legend(fontsize="small")
    
    plt.tight_layout()
    plt.show()





## Question 3


def run_Q3_time():
    """
    Q3: Predator–prey time history with RK8.

    Inputs: none (params inside function)
    Outputs: time plot of prey (green) and predator (magenta)
    """
    # Params and initial populations
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.5, 0.4

    # RK8 solver
    t_rk_pp, N1_rk_pp, N2_rk_pp = solve_rk8(
        dNdt_predprey, N1_init, N2_init,
        dT=2.0, t_final=50, a=a, b=b, c=c, d=d
    )

    # Plot populations over time
    plt.figure(figsize=(9,6))
    plt.plot(t_rk_pp, N1_rk_pp, 'g', label='Prey N1')
    plt.plot(t_rk_pp, N2_rk_pp, 'm', label='Predator N2')

    # Mark max points
    plt.scatter([t_rk_pp[np.argmax(N1_rk_pp)]], [np.max(N1_rk_pp)],
                s=50, c='g', marker='o', label='Prey peak')
    plt.scatter([t_rk_pp[np.argmax(N2_rk_pp)]], [np.max(N2_rk_pp)],
                s=50, c='m', marker='o', label='Predator peak')

    plt.xlabel('Time (years)')
    plt.ylabel('Population (normalized)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()




def run_Q3_phase():
    """
    Q3: Predator–prey phase diagram with RK8.

    Inputs: none (params inside function)
    Outputs: phase plot with trajectory, start, equilibrium,
             nullclines, and vector field
    """
    # Params and initial populations
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.5, 0.4

    # RK8 solver, included a step size of 0.1 to see a smoother curve
    t_rk_pp, N1_rk_pp, N2_rk_pp = solve_rk8(
        dNdt_predprey, N1_init, N2_init,
        dT=0.1, t_final=50, a=a, b=b, c=c, d=d
    )

    plt.figure(figsize=(6,6))
    # Trajectory path
    plt.plot(N1_rk_pp, N2_rk_pp, 'k-', label='Trajectory')
    plt.scatter([N1_rk_pp[0]], [N2_rk_pp[0]], s=40, marker='o', label='Start')

    # Equilibrium point, labeling it as a star here
    N1_star, N2_star = c/d, a/b
    plt.scatter([N1_star], [N2_star], c='gold', s=100, marker='*', label='Equilibrium')

    # Nullcline lines
    plt.axvline(N1_star, linestyle='--', color='red', linewidth=1, label='N1 nullcline')
    plt.axhline(N2_star, linestyle='--', color='blue', linewidth=1, label='N2 nullcline')

    # Vector field grid
    nx = ny = 20
    x = np.linspace(0.01, max(N1_rk_pp)*1.2, nx)
    y = np.linspace(0.01, max(N2_rk_pp)*1.2, ny)
    X, Y = np.meshgrid(x, y)
    U = a*X - b*X*Y
    V = -c*Y + d*X*Y
    # Normalize arrows
    speed = np.hypot(U, V)
    U, V = U/(speed+1e-12), V/(speed+1e-12)

    # Used quiver instead of zip, didnt feel like quiver was necessary looping through each point
    plt.quiver(X, Y, U, V, alpha=0.3, angles='xy', scale_units='xy', scale=12)

    plt.xlabel('Prey N1')
    plt.ylabel('Predator N2')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



