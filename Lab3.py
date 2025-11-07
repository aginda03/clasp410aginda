#!/usr/bin/env python3
"""
Lab 3 – Heat Equation and Permafrost Modeling

This script solves the 1D heat equation under different boundary conditions (Dirichlet and Neumann)
and extends it to model permafrost temperature changes over time in Greenland. 
It also runs warming scenarios to see how rising surface temperatures affect active layer depth and permafrost stability.
"""


import numpy as np
import matplotlib.pyplot as plt

# Sets solver functions up before answering questions
# Does not have a "to replicate my results:" section, possibly in lab report.
# Heat solvers are separated into Dirichlet and Neumann boundary conditions
# Might be compressible? This code is very chunky, would be nice to have less.


def solve_heat(func0, funcub, funclb, xstop=1, tstop=0.2,
               dx=0.02, dt=0.0002, c2=1, n=False):
    """
    Runs a forward-difference heat solver with fixed ends (Dirichlet BCs).

    Parameters
    ----------
    func0 : function
        babdsbbadbbds # FIX LATER
    funcub : function
        absdhbsabdhasb # FIX LATER
    funclb : function
        ansdnhshdsdah # FIX LATER
    xstop : float
        Length of the domain (m).
    tstop : float
        Total simulation time (s).
    dx : float
        Spatial step size (m).
    dt : float
        Time step (s).
    c2 : float
        Thermal diffusivity (m²/s).
    n : boolean, defaults to False
        True if Neumann boundary conditions applied, False if Dirichlet.

    Returns
    -------
    U : ndarray
        Temperature field [space * time].
    x : ndarray
        Spatial grid (m).
    t : ndarray
        Time grid (s).
    """
    # Check that dt isn’t too big or it’ll go unstable
    dt_max = dx**2 / (2 * c2)
    if dt > dt_max:
        raise ValueError(f"peligroso: dt {dt} must be <= dt_max {dt_max}")

    # set up number of points and grids
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # temp array and starting shape (warm middle, cold ends in ex. sol)
    U = np.zeros((M, N))
    U[:, 0] = func0(x)
    # Dirichlet boundary conditions (ends suspended in 0C ice in ex. sol)
    if not n:
        U[0, :] = funcub(t)
        U[M-1, :] = funclb(t)

    r = c2 * (dt / dx**2)

    # main time loop
    for j in range(N - 1):
        U[1:M-1, j+1] = (1 - 2*r) * U[1:M-1, j] + r * (U[2:M, j] + U[:M-2, j])
        # Check for Neumann conditions:
        if n:
            U[0, j+1] = U[1, j+1]       # left end: zero gradient
            U[M-1, j+1] = U[M-2, j+1]   # right end: zero gradient

    return U, x, t


# Initial and boundary condition functions to pass into solver:
# rod_diri_0 produces the initial temps on the rod spatially
def rod_diri_0(x): return (4 * x - 4 * x ** 2)
# rod_diri_ub and rod_diri_lb produce end-suspended-in-ice
def rod_diri_ub(t): return np.zeros(t.size)
def rod_diri_lb(t): return np.zeros(t.size)


def plot_heatsolve_dirichlet(xstop=1, tstop=0.2, dx=0.02, dt=0.0002, c2=1):
    """
    Solves the heat equation (Dirichlet BCs) and makes a 2D contour plot.

    Inputs are the same as the solver, and it just displays a figure of
    temperature vs. time and position.

    run with plot_heatsolve_dirichlet()
    """    
    # run the solver and set up grids for plotting
    U, x, t = solve_heat(rod_diri_0, rod_diri_ub, rod_diri_lb, xstop, tstop,
                         dx, dt, c2, n=False)
    T, X = np.meshgrid(t, x, indexing="ij")

    # make the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.pcolor(T, X, U.T, cmap="inferno")

    # add labels and colorbar
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.15)
    cbar.set_label(r"Temperature $T\;(^\circ\mathrm{C})$")
    ax.set_xlabel(r"Time $t\;(\mathrm{s})$")
    ax.set_ylabel(r"Position $x\;(\mathrm{m})$")
    ax.set_title(r"Heat Equation (Dirichlet BCs)")

    # clean layout and show it
    plt.tight_layout()
    plt.show()


def plot_heatsolve_neumann(xstop=1, tstop=0.2, dx=0.02, dt=0.0002, c2=1):
    """
    Solves the heat equation (Neumann BCs) and plots the 2D contour.

    Inputs are the same as the solver, outputs a figure with labeled axes
    showing how temp spreads when the ends are insulated.

    run with plot_heatsolve_neumann(), NOTE: again same here just for the hw assignment
    """
    # run the solver and set up grids for plotting
    U, x, t = solve_heat(rod_diri_0, rod_diri_ub, rod_diri_lb, xstop, tstop,
                         dx, dt, c2, n=True)
    T, X = np.meshgrid(t, x, indexing="ij")

    # make the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.pcolor(T, X, U.T, cmap="inferno")

    # add labels and colorbar
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.15)
    cbar.set_label(r"Temperature $T\;(^\circ\mathrm{C})$")
    ax.set_xlabel(r"Time $t\;(\mathrm{s})$")
    ax.set_ylabel(r"Position $x\;(\mathrm{m})$")
    ax.set_title(r"Heat Equation (Neumann BCs)")

    # clean layout and show it
    plt.tight_layout()
    plt.show()


'''
NOTE : for the hw assingment, just to check energy conservation
# Just checking, energy in - energy out

U, x, t = solve_heat_neumann(dx=0.02, dt=0.0002)
energy = np.trapezoid(U, x, axis=0)  # integrate over x at each time
print("Energy change:", energy[-1] - energy[0])


U, x, t = solve_heat_dirichlet(dx=0.02, dt=0.0002)
energy = np.trapezoid(U, x, axis=0)
print("Energy change:", energy[-1] - energy[0])
'''

# ==========================================================
# Homework N-D


def plot_comparison(dx=0.02, dt=0.0002, xstop=1, tstop=0.2, c2=1):
    """
    Runs both solvers (Dirichlet + Neumann) and compares them side by side.

    Parameters
    ----------
    dx, dt, xstop, tstop, c2 : float
        Same as above, controlling grid spacing, time step, domain, etc.

    Returns
    -------
    None
        Just shows a figure with both cases for visual comparison.

        run with plot_comparison()
    """
    # run both solvers to compare fixed vs insulated ends
    U_dir, x, t = solve_heat(rod_diri_0, rod_diri_ub, rod_diri_lb, xstop,
                             tstop, dx, dt, c2, n=False)
    U_neu, _, _ = solve_heat(rod_diri_0, rod_diri_ub, rod_diri_lb, xstop,
                             tstop, dx, dt, c2, n=True)

    # set up grids and color scale
    vmin, vmax = 0, max(U_dir.max(), U_neu.max())
    T, X = np.meshgrid(t, x, indexing="ij")

    # make side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axs[0].pcolor(T, X, U_dir.T, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    axs[0].set_title("Dirichlet (Fixed Ends)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (m)")

    im2 = axs[1].pcolor(T, X, U_neu.T, cmap="inferno", vmin=vmin, vmax=vmax,
                        shading="auto")
    axs[1].set_title("Neumann (Insulated Ends)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Position (m)")

    # adjust layout and colorbar
    plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.25)
    cbar = fig.colorbar(im2, ax=axs, orientation="horizontal", fraction=0.05,
                        pad=0.1)
    cbar.set_label("Temperature (°C)")

    # add a main title and show it
    fig.suptitle("Heat Equation: Dirichlet vs Neumann Boundary Conditions",
                 y=0.98, fontsize=12)
    plt.show()


'''
With Dirichlet conditions, the wire ends stay at 0°C, so heat flows out until
the whole wire cools down. With Neumann conditions, the ends are insulated,
meaning no heat escapes and the wire slowly evens out but stays warm overall.
The Neumann case represents a wire with insulated ends where there’s no heat
transfer to the surroundings.
'''
# ==========================================================

# QUESTION 1: Validation of Heat Solver


def validate_heat_solver():
    """
    Checks that the heat solver works by running the validation setup
    from the lab handout and printing a little table of results.

    Returns
    -------
    U_valid : ndarray
        Validation temperature field.
    x_valid : ndarray
        Grid (m).
    t_valid : ndarray
        Time (s).

        run with validate_heat_solver()
    """

    # solve the validation case using the Dirichlet solver
    U_valid, x_valid, t_valid = solve_heat(rod_diri_0, rod_diri_ub,
                                           rod_diri_lb, xstop=1, tstop=0.2,
                                           dx=0.2, dt=0.02, c2=1)

    # print a chunk of the result table to compare with the lab handout
    print("\nValidation Table (approx values):")
    print("i\\j\t" + "\t".join([f"{j}" for j in range(0, 11)]))

    # loop over each spatial row (i) and print first 10 time steps (j)
    # shows how the temperature changes over time at each position
    for i in range(U_valid.shape[0]):
        row = "\t".join([f"{U_valid[i, j]:.4f}" for j in range(0, 11)])
        print(f"{i}\t{row}")

    # make a color plot to see how heat spreads over time
    fig, ax = plt.subplots(figsize=(6, 4))
    T, X = np.meshgrid(t_valid, x_valid, indexing="ij")
    c = ax.pcolor(T, X, U_valid.T, cmap="inferno", vmin=0, vmax=1, 
                  shading="auto")

    # add colorbar and axis labels for clarity
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.15)
    cbar.set_label(r"Temperature $T\;(^\circ\mathrm{C})$")
    ax.set_xlabel(r"Time $t\;(\mathrm{s})$")
    ax.set_ylabel(r"Position $x\;(\mathrm{m})$")
    ax.set_title(r"Validation: Forward-Difference Heat Solver (Dirichlet BCs)")

    # clean layout and show figure
    plt.tight_layout()
    plt.show()

    # return arrays in case we want to use them later
    return U_valid, x_valid, t_valid


def compare_to_reference():
    """
    Compares the validation output to the given reference dataset and
    plots the absolute error.

    Returns
    -------
    Nada
        Just prints errors and shows a color map of differences.
    """

    # import the reference solution we’re comparing against
    # Concern here - if saved as a library,
    # install details need to be outlined in this file
    from test_solution import sol10p3

    # run the solver validation automatically
    U_valid, x_valid, t_valid = validate_heat_solver()

    print("\n--- Comparing to Reference Solution ---")
    print("Solver output shape:", U_valid.shape)
    print("Reference solution shape:", sol10p3.shape)

    # make sure both arrays are the same size before comparing
    min_m = min(U_valid.shape[0], sol10p3.shape[0])
    min_n = min(U_valid.shape[1], sol10p3.shape[1])
    U_sub = U_valid[:min_m, :min_n]

    # calculate absolute difference between our solver and the reference
    abs_diff = np.abs(U_sub - sol10p3)

    # basic error stats so we know how close it is, threw these in the report
    max_error = abs_diff.max()
    mean_error = abs_diff.mean()
    print(f"\nMax error: {max_error:.4e}")
    print(f"Mean absolute error: {mean_error:.4e}")

    # heamap of the errors
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(abs_diff, cmap="plasma", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="|Numerical - Reference| (°C)")
    ax.set_title("Absolute Error vs. Reference Solution")
    ax.set_xlabel("Time index j")
    ax.set_ylabel("Position index i")

    plt.tight_layout()
    plt.show()


# QUESTION 2: Kangerlussuaq Permafrost Model

# ---- Surface temperature function ----


t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                     10.7, 8.5, 3.1, -6.0, -12.0, -16.9])


def temp_kanger(t_days):
    """
    Gives Kangerlussuaq’s surface temperature (°C) for a given time in days, gets called in the permafrost solver.

    Parameters
    ----------
    t_days : array-like
        Time in days through the year.

    Returns
    -------
    temps : ndarray
        Surface temp values (°C).
    """
    # find the amplitude of the yearly temp swing (how much it varies)
    t_amp = (t_kanger - t_kanger.mean()).max()

    # make a smooth sine wave to model yearly temperature change
    return t_amp * np.sin(np.pi / 180 * t_days - np.pi / 2) + t_kanger.mean()


def solve_permafrost(xstop=100, years=200, dx=0.5, dt_days=0.5, c2=0.25e-6):
    """
    Solves 1D heat diffusion in 100 m of soil with surface temps, gets called in the permafrost runner.

    Parameters
    ----------
    xstop : float
        Depth (m).
    years : int
        Total years simulated.
    dx : float
        Depth step (m).
    dt_days : float
        Time step (days).
    c2 : float
        Thermal diffusivity (m²/s).

    Returns
    -------
    U : ndarray
        Temperature field [depth × time].
    x : ndarray
        Depth grid (m).
    t : ndarray
        Time grid (days).
    """
    # convert everything from days to seconds
    sec_per_day = 24 * 3600
    dt = dt_days * sec_per_day
    tstop = years * 365 * sec_per_day

    # check for stability so the solver doesn’t blow up
    dt_max = dx**2 / (2 * c2)
    if dt > dt_max:
        raise ValueError(f"Unstable! dt={dt:.2e} s > dt_max={dt_max:.2e} s")

    # set up grid sizes and arrays
    N = int(tstop / dt)
    M = int(xstop / dx) + 1
    t = np.linspace(0, years * 365, N)  # time in days
    x = np.linspace(0, xstop, M)
    U = np.zeros((M, N))
    r = c2 * dt / dx**2

    # step through time to update temps
    for j in range(N - 1):
        U[1:M-1, j+1] = (1 - 2*r) * U[1:M-1, j] + r * (U[2:M, j] + U[:M-2, j])
        U[0, j+1] = temp_kanger(t[j] % 365)   # surface temp cycles yearly
        U[M-1, j+1] = 5.0                     # bottom held at 5°C

    return U, x, t


def run_permafrost(show_plots=True, years=200):
    """
    Runs the permafrost model and optionally plots results (100 m soil).

    Parameters
    ----------
    show_plots : bool
        Show figures if True.
    years : int
        Duration of the run.

    Returns
    -------
    U_p : ndarray
        Temperature field.
    x_p : ndarray
        Depth grid.
    t_p : ndarray
        Time grid (days).
    active_depth : float
        Depth of seasonal thaw (m).
    permafrost_base : float
        Depth of frozen layer base (m).

        run with run_permafrost()
    """
    # run the permafrost simulation
    U_p, x_p, t_p = solve_permafrost(xstop=100, years=years, dx=0.5, 
                                     dt_days=0.5)

    # get data from the final simulated year
    loc = int(-365 / 0.5)  # last 365 days (since dt=0.5 days)
    winter = U_p[:, loc:].min(axis=1)  # coldest temps each depth
    summer = U_p[:, loc:].max(axis=1)  # warmest temps each depth
    mean_bottom_temp = U_p[-1, loc:].mean()
    mean_surface_temp = U_p[0, loc:].mean()

    # helper function to find where temp crosses a target value (like 0°C)
    def find_depth(profile, depth, target=0):
        idx = np.argwhere(np.diff(np.sign(profile - target))).flatten()
        return depth[idx[0]] if len(idx) > 0 else np.nan

    # find how deep the seasonal thaw goes and where the permafrost starts
    active_depth = find_depth(summer, x_p)
    permafrost_base = find_depth(winter, x_p)

    # print out all the key numbers
    print("\n===== PERMAFROST MODEL RESULTS =====")
    print(f"Simulation duration: {years} years")
    print(f"Mean surface temperature (final year): {mean_surface_temp:.2f} °C")
    print(f"Mean bottom temperature (final year): {mean_bottom_temp:.2f} °C")
    print(f"Estimated active-layer depth: {active_depth:.2f} m")
    print(f"Estimated permafrost base depth: {permafrost_base:.2f} m")
    print(f"Temperature range at surface: {summer[0]:.1f} to {winter[0]:.1f} °C")
    print(f"Temperature range at 100 m depth: {summer[-1]:.2f} to {winter[-1]:.2f} °C")
    print("====================================")

    # make the plots if the flag is on 
    if show_plots:
        from matplotlib.colors import TwoSlopeNorm

        # first plot shows how temperature changes through the ground over time
        # x-axis is time in years, y-axis is depth
        # (0 m = surface, 100 m = deep soil)
        # color shows temperature, where red is warmer and blue is colder
        fig, ax = plt.subplots(figsize=(7, 4))

        # convert time from days to years
        T_yrs, X_m = np.meshgrid(t_p / 365, x_p, indexing="ij")

        # center the color map around 0°C so freezing is easy to see
        norm_full = TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6)
        c = ax.pcolormesh(T_yrs, X_m, U_p.T, cmap="coolwarm",
                          norm=norm_full, shading="auto")

        # flip the y-axis so the surface (0 m) is at the top
        ax.invert_yaxis()
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Permafrost Temperature Evolution")

        # add colorbar so we know what the colors mean
        plt.colorbar(c, ax=ax, label="Temperature (°C)")
        plt.tight_layout()
        plt.show()

        # second plot does basically the same thing but zooms in near freezing
        # this helps visualize where the soil thaws and freezes over time
        # NOTE: just used the first in the report, 
        # ended up just changing limits there
        fig, ax_zoom = plt.subplots(figsize=(7, 4))
        norm = TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6)
        c_zoom = ax_zoom.pcolormesh(T_yrs, X_m, U_p.T, cmap="coolwarm",
                                    norm=norm, shading="auto")

        # draw a black contour line where temperature = 0°C (freezing point)
        cs = ax_zoom.contour(T_yrs, X_m, U_p.T, levels=[0],
                             colors="black", linewidths=1.2)
        ax_zoom.clabel(cs, fmt="0 °C", colors="black")

        # again, surface on top and depth going down
        ax_zoom.invert_yaxis()
        ax_zoom.set_xlabel("Time (years)")
        ax_zoom.set_ylabel("Depth (m)")
        ax_zoom.set_title("Permafrost Temperature Evolution")
        plt.colorbar(c_zoom, ax=ax_zoom, label="Temperature (°C)")
        plt.tight_layout()
        plt.show()

        # third plot shows temperature vs. depth
        # for just the final simulated year
        # summer = red line, winter = blue line
        # the gap between them near the surface
        # is the active layer that freezes/thaws
        fig, ax2 = plt.subplots(figsize=(5, 6))
        ax2.plot(summer, x_p, label="Summer", color="red")
        ax2.plot(winter, x_p, label="Winter", color="blue")

        # add a dashed line at 0°C for reference
        ax2.axvline(0, color="k", lw=1, ls="--")

        # invert y-axis again so top = surface
        ax2.invert_yaxis()
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Depth (m)")
        ax2.set_title("Ground Temperature Profiles – Final Year")

        # label the lines and show everything
        ax2.legend()
        plt.tight_layout()
        plt.show()

    # return the arrays and layer depths so we can use them later
    return U_p, x_p, t_p, active_depth, permafrost_base


# QUESTION 3: Global Warming Scenarios

def run_warming_scenarios(temp_shifts=(0.0, 0.5, 1.0, 3.0), years=200, 
                          show_plots=True):
    """
    Runs the permafrost sim multiple times under different warming offsets.

    Parameters
    ----------
    temp_shifts : tuple
        Surface warming values (°C) to test.
    years : int
        Years per run.
    show_plots : bool
        If True, show comparison plot of final-year summer profiles.

    Returns
    -------
    results : list of tuples
        Each entry is (ΔT, active_layer_depth, permafrost_base_depth).

        run with run_warming_scenarios()
    """
    # list to store all the simulation results
    results = []

    # loop through each temperature increase we want to test
    for dT in temp_shifts:
        # make a new version of the surface temperature 
        # function that’s slightly warmer
        def temp_kanger_shifted(t_days):
            t_amp = (t_kanger - t_kanger.mean()).max()
            return t_amp * np.sin(np.pi / 180 * t_days - np.pi / 2) + \
                t_kanger.mean() + dT

        # temporarily replace the old surface function with the shifted one
        global temp_kanger
        original_temp_kanger = temp_kanger
        temp_kanger = temp_kanger_shifted

        # run the permafrost simulation for this warming level (no plots yet)
        print(f"\n=== Running simulation with +{dT:.1f} °C surface warming ===")
        U_p, x_p, t_p, active, base = run_permafrost(show_plots=False, 
                                                     years=years)

        # store the results so we can print them all together later
        results.append((dT, active, base))

        # restore the original surface temp function so next run starts clean
        temp_kanger = original_temp_kanger

    # print out a nice summary table comparing all the runs
    print("\n===== GLOBAL WARMING SCENARIO RESULTS =====")
    print(f"{'ΔT (°C)':>8} | {'Active Layer (m)':>17} | {'Permafrost Base (m)':>21}")
    print("-" * 52)
    for dT, active, base in results:
        print(f"{dT:8.1f} | {active:17.2f} | {base:21.2f}")
    print("===========================================")

    # make a plot comparing the summer temperature profiles 
    # under different warming cases
    if show_plots:
        colors = ["gold", "orange", "red", "darkred"]
        fig, ax = plt.subplots(figsize=(5, 6))

        # loop through each scenario and plot the final-year summer profile
        for (dT, _, _), color in zip(results, colors):
            # re-create the shifted surface function for new temperature offset
            def temp_kanger_shifted(t_days):
                t_amp = (t_kanger - t_kanger.mean()).max()
                return t_amp * np.sin(np.pi / 180 * t_days - np.pi / 2) \
                    + t_kanger.mean() + dT
            temp_kanger = temp_kanger_shifted

            # run solver again to get temperature field
            U_p, x_p, t_p = solve_permafrost(years=years)

            # take only the final year to get the 
            # summer and winter temperature ranges
            loc = int(-365 / 0.5)
            winter = U_p[:, loc:].min(axis=1)
            summer = U_p[:, loc:].max(axis=1)

            # plot the summer profile for this warming case
            ax.plot(summer, x_p, label=f"Summer +{dT}°C", color=color)

        # restore the original surface function again
        temp_kanger = original_temp_kanger

        # add the 0°C line and clean up the plot
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Summer Profiles – Global Warming Scenarios")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # return everything collected
    return results
