#!/usr/bin/env python3
'''
Lab 2: Population Control
Andrew Inda

This script models Lab 2, solves Lotka–Volterra equations
for competition and predator–prey systems. Both Euler and
RK8 solvers are used.

%run Lab2.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 
from matplotlib.lines import Line2D

plt.style.use("seaborn-v0_8")  # nicer looking plots, not the default ugly matplotlib


## Derivative functions


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    """
    Competition model ODEs. Two species competing.
    """
    # unpack the species populations
    N1, N2 = N
    # logistic growth for each, minus the part where they fight each other
    dN1dt = a * N1 * (1 - N1) - b * N1 * N2
    dN2dt = c * N2 * (1 - N2) - d * N1 * N2
    return [dN1dt, dN2dt]


def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    """
    Predator–prey model ODEs. Prey growth and predator hunting them.
    """
    # grab the two species
    N1, N2 = N
    # prey grows on its own but gets eaten
    dN1dt = a * N1 - b * N1 * N2
    # predator dies if no food, but grows if it eats
    dN2dt = -c * N2 + d * N1 * N2
    return [dN1dt, dN2dt]


## Euler solver


def euler_solve(func, N1_init=0.3, N2_init=0.6, dT=0.1, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    Euler solver (fixed step). Models the populations step by step.
    """
    # build time array manually (fixed step size)
    time = np.arange(0, t_final + dT, dT)
    # arrays to keep populations over time
    N1 = np.zeros_like(time)
    N2 = np.zeros_like(time)
    # start values
    N1[0], N2[0] = N1_init, N2_init

    # loop over time, classic Euler method
    for i in range(1, len(time)):
        dN1dt, dN2dt = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        # Euler update = old value + slope * step
        N1[i] = N1[i-1] + dT * dN1dt
        N2[i] = N2[i-1] + dT * dN2dt
    return time, N1, N2


## RK8 solver


def solve_rk8(func, N1_init=0.3, N2_init=0.6, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    RK8 solver (DOP853, adaptive step).
    """
    # scipy’s built-in solver — does the heavy lifting for us
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=(a, b, c, d), method="DOP853", max_step=dT)
    # result.y is 2D, first row = N1, second row = N2
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
    """
    # base parameters
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.3, 0.6

    # Euler version
    t_euler, N1_euler, N2_euler = euler_solve(
        dNdt_comp, N1_init, N2_init, dT=0.1, t_final=100, a=a, b=b, c=c, d=d)

    # RK8 version
    t_rk8, N1_rk8, N2_rk8 = solve_rk8(
        dNdt_comp, N1_init, N2_init, dT=10, t_final=100, a=a, b=b, c=c, d=d)

    # overlay both to see difference
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
    """
    # predator-prey baseline params
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.3, 0.6
    t_final = 100

    # Euler with small dt to stay somewhat accurate
    t_euler, N1_euler, N2_euler = euler_solve(
        dNdt_predprey, N1_init, N2_init,
        dT=0.05, t_final=t_final,
        a=a, b=b, c=c, d=d
    )

    # RK8 with adaptive stepping
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_predprey, N1_init, N2_init,
        dT=5.0, t_final=t_final,
        a=a, b=b, c=c, d=d
    )

    # compare visually
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

def run_Q2_par_sweep(t_final=20):
    """
    Sweep parameters a, b, c, d individually for competition model.
    """
    base_params = {"a": 1, "b": 2, "c": 1, "d": 3}
    N1_init, N2_init = 0.5, 0.5

    # different values to try for each parameter
    sweep_ranges = {
        "a": [0.5, 1, 1.5, 2],
        "b": [0.5, 1, 2, 3],
        "c": [0.5, 1, 1.5, 2],
        "d": [0.5, 1, 2, 3]
    }

    # set up 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # loop over each param sweep
    for idx, param in enumerate(["a", "b", "c", "d"]):
        ax = axes[idx]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_ranges[param])))
        param_handles = []

        for val, col in zip(sweep_ranges[param], colors):
            params = base_params.copy()
            params[param] = val

            # run solver with that param
            t, N1, N2 = solve_rk8(
                dNdt_comp,
                N1_init=N1_init, N2_init=N2_init,
                t_final=t_final, dT=2,
                a=params["a"], b=params["b"], c=params["c"], d=params["d"]
            )

            # dashed for N1, solid for N2
            h1, = ax.plot(t, N1, "--", color=col)
            h2, = ax.plot(t, N2, "-",  color=col)
            param_handles.append((h1, f"{param}={val}"))

        # per subplot legend showing the sweep values
        ax.legend(
            handles=[h for h, _ in param_handles],
            labels=[lbl for _, lbl in param_handles],
            title=f"Sweep {param}",
            loc="lower right",
            fontsize="x-small",
            frameon=False
        )

        ax.set_title(f"Sweep {param}")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Population")

    # one global legend for dashed vs solid = which species
    species_lines = [
        Line2D([], [], color="k", linestyle="--", label="N1 (species 1)"),
        Line2D([], [], color="k", linestyle="-",  label="N2 (species 2)")
    ]
    fig.legend(
        handles=species_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize="medium"
    )

    fig.suptitle("Competition Model Parameter Sweeps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.show()


def run_Q2_init_sweep(t_final=20):
    """
    Sweep initial conditions for N1 and N2 separately.
    """
    base_params = {"a": 1, "b": 2, "c": 1, "d": 3}

    # values to test, just chose them arbitrarily
    sweep_N1 = [0.1, 0.3, 0.5, 0.9]
    sweep_N2 = [0.1, 0.3, 0.5, 0.9]
    fixed_val = 0.5  # hold one constant while changing the other

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # left: vary N1 init
    colors = plt.cm.plasma(np.linspace(0, 1, len(sweep_N1)))
    for n1_init, col in zip(sweep_N1, colors):
        t, N1, N2 = solve_rk8(
            dNdt_comp,
            N1_init=n1_init, N2_init=fixed_val,
            t_final=t_final, dT=2,
            a=base_params["a"], b=base_params["b"],
            c=base_params["c"], d=base_params["d"]
        )
        axes[0].plot(t, N1, "--", color=col)
        axes[0].plot(t, N2, "-",  color=col, label=f"N1={n1_init}")
    axes[0].set_title("Sweep N1")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Population")
    axes[0].legend(fontsize="x-small", frameon=False)

    # right: vary N2 init
    colors = plt.cm.plasma(np.linspace(0, 1, len(sweep_N2)))
    for n2_init, col in zip(sweep_N2, colors):
        t, N1, N2 = solve_rk8(
            dNdt_comp,
            N1_init=fixed_val, N2_init=n2_init,
            t_final=t_final, dT=2,
            a=base_params["a"], b=base_params["b"],
            c=base_params["c"], d=base_params["d"]
        )
        axes[1].plot(t, N1, "--", color=col)
        axes[1].plot(t, N2, "-",  color=col, label=f"N2={n2_init}")
    axes[1].set_title("Sweep N2")
    axes[1].set_xlabel("Time (years)")
    axes[1].legend(fontsize="x-small", frameon=False)

    # global dashed vs solid legend
    species_lines = [
        Line2D([], [], color="k", linestyle="--", label="N1 (species 1)"),
        Line2D([], [], color="k", linestyle="-",  label="N2 (species 2)")
    ]
    fig.legend(
        handles=species_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
        fontsize="medium"
    )

    fig.suptitle("Initial Condition Sweeps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


## Question 3 — predator–prey sweeps

def run_Q3_sweep_N1(t_final=50):
    """
    Predator–prey dynamics sweeping N1 initial.
    Two subplots: time histories (with dashed/solid legend) and phase diagram (with sweep legend on right).
    """
    a, b, c, d = 1, 2, 1, 3
    sweep_vals = [0.2, 0.4, 0.6, 0.8]   # different initial prey populations
    N2_fixed = 0.4                      # predator held constant

    # make two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_vals)))

    for n1_init, col in zip(sweep_vals, colors):
        # run model for each starting N1
        t, N1, N2 = solve_rk8(
            dNdt_predprey,
            N1_init=n1_init, N2_init=N2_fixed,
            dT=0.2, t_final=t_final,
            a=a, b=b, c=c, d=d
        )
        # dashed = prey, solid = predator
        axes[0].plot(t, N1, "--", color=col)
        axes[0].plot(t, N2, "-",  color=col)
        # phase diagram = trajectory of populations
        axes[1].plot(N1, N2, color=col, label=f"N1={n1_init}")

    # labels for time history plot
    axes[0].set_title("Sweep N1 Initial: Time Histories")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Population")

    # labels for phase plot
    axes[1].set_title("Sweep N1 Initial: Phase Diagram")
    axes[1].set_xlabel("Prey N1")
    axes[1].set_ylabel("Predator N2")

    # equilibrium point + nullclines (fancy way of saying "where slopes = 0")
    N1_star, N2_star = c/d, a/b
    axes[1].scatter([N1_star], [N2_star], c="gold", s=100, marker="*", label="Equilibrium")
    axes[1].axvline(N1_star, linestyle="--", color="red", linewidth=1, label="N1 nullcline")
    axes[1].axhline(N2_star, linestyle="--", color="blue", linewidth=1, label="N2 nullcline")

    # legend for sweep values (right side, phase diagram)
    axes[1].legend(
        title="Initial Prey",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="x-small",
        frameon=False
    )

    # dashed vs solid legend (time history plot only)
    species_lines = [
        Line2D([], [], color="k", linestyle="--", label="Prey N1"),
        Line2D([], [], color="k", linestyle="-",  label="Predator N2")
    ]
    axes[0].legend(handles=species_lines, loc="upper right", fontsize="small", frameon=False)

    fig.suptitle("Predator–Prey Dynamics: Sweep N1 Initial", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # leaving room for outside legend
    plt.show()


def run_Q3_sweep_N2(t_final=50):
    """
    Predator–prey dynamics sweeping N2 initial.
    Two subplots: time histories (with dashed/solid legend) and phase diagram (with sweep legend on right).
    """
    a, b, c, d = 1, 2, 1, 3
    sweep_vals = [0.2, 0.4, 0.6, 0.8]   # different initial predator pops
    N1_fixed = 0.5                      # prey held constant

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(sweep_vals)))

    for n2_init, col in zip(sweep_vals, colors):
        # run model for each starting N2
        t, N1, N2 = solve_rk8(
            dNdt_predprey,
            N1_init=N1_fixed, N2_init=n2_init,
            dT=0.2, t_final=t_final,
            a=a, b=b, c=c, d=d
        )
        axes[0].plot(t, N1, "--", color=col)
        axes[0].plot(t, N2, "-",  color=col)
        axes[1].plot(N1, N2, color=col, label=f"N2={n2_init}")

    # left plot labels
    axes[0].set_title("Sweep N2 Initial: Time Histories")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Population")

    # right plot labels
    axes[1].set_title("Sweep N2 Initial: Phase Diagram")
    axes[1].set_xlabel("Prey N1")
    axes[1].set_ylabel("Predator N2")

    # equilibrium + nullclines again
    N1_star, N2_star = c/d, a/b
    axes[1].scatter([N1_star], [N2_star], c="gold", s=100, marker="*", label="Equilibrium")
    axes[1].axvline(N1_star, linestyle="--", color="red", linewidth=1, label="N1 nullcline")
    axes[1].axhline(N2_star, linestyle="--", color="blue", linewidth=1, label="N2 nullcline")

    # legend for sweep values (on right of phase diagram)
    axes[1].legend(
        title="Initial Predator",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="x-small",
        frameon=False
    )

    # dashed/solid legend for time histories
    species_lines = [
        Line2D([], [], color="k", linestyle="--", label="Prey N1"),
        Line2D([], [], color="k", linestyle="-",  label="Predator N2")
    ]
    axes[0].legend(handles=species_lines, loc="upper right", fontsize="small", frameon=False)

    fig.suptitle("Predator–Prey Dynamics: Sweep N2 Initial", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()


# quick reminder of what the baseline params mean:
#   a = prey growth rate (how fast prey breed)
#   b = predation rate (how good predators are at catching prey)
#   c = predator death rate (how fast predators die off if no food)
#   d = predator growth rate (how much predators gain from eating prey)
#
# baseline initial conditions:
#   N1_init = 0.5   (prey start halfway up)
#   N2_init = 0.4   (predators a bit less)
#
# equilibrium point for these numbers:
#   N1* = c/d = 1/3 ≈ 0.33
#   N2* = a/b = 0.5



def run_Q3_sweep_init(t_final=50):
    """
    Predator–prey dynamics: sweep a, b, c, d.
    Four subplots in a 2x2 grid with sweep legends on right
    and a global dashed/solid legend for species at the bottom.
    """
    # baseline params
    base = {"a": 1.0, "b": 2.0, "c": 1.0, "d": 3.0}
    N1_init, N2_init = 0.5, 0.4

    # ranges for each param
    sweeps = {
        "a": [0.5, 1.0, 1.5, 2.0],   # prey growth
        "b": [0.5, 1.0, 2.0, 3.0],   # predation rate
        "c": [0.5, 1.0, 1.5, 2.0],   # predator death rate
        "d": [0.5, 1.0, 2.0, 3.0]    # predator growth rate
    }
    nice = {"a": "Prey growth (a)", "b": "Predation rate (b)",
            "c": "Predator loss (c)", "d": "Predator gain (d)"}

    # 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, param in enumerate(["a", "b", "c", "d"]):
        ax = axes[idx]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sweeps[param])))

        for val, col in zip(sweeps[param], colors):
            # copy baseline, change one param
            p = base.copy()
            p[param] = val

            # run solver with this param
            t, N1, N2 = solve_rk8(
                dNdt_predprey,
                N1_init=N1_init, N2_init=N2_init,
                dT=0.2, t_final=t_final,
                a=p["a"], b=p["b"], c=p["c"], d=p["d"]
            )
            ax.plot(t, N1, "--", color=col)
            ax.plot(t, N2, "-",  color=col, label=f"{param}={val:g}")

        ax.set_title(f"Sweep {nice[param]}")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Population")

        # legend for sweep values sits on the right side of each subplot
        ax.legend(
            title=f"{param} values",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="x-small",
            frameon=False
        )

    # one global legend for dashed vs solid (applies to all subplots)
    species_lines = [
        Line2D([], [], color="k", linestyle="--", label="Prey N1 (dashed)"),
        Line2D([], [], color="k", linestyle="-",  label="Predator N2 (solid)")
    ]
    fig.legend(handles=species_lines,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.90),
               ncol=2,
               frameon=False,
               fontsize="medium")

    fig.suptitle("Predator–Prey Parameter Sweeps (a, b, c, d)", fontsize=16)
    plt.tight_layout(rect=[0, 0.08, 0.9, 0.95])  # make space at bottom + right
    plt.show()



