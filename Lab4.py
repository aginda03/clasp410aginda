##!/usr/bin/env python3

"""
Lab 4  Universality: Sick of Burning, Andrew Inda
This script simulates how things like forest fires or diseases spread across a grid using simple probability rules. 
It runs different experiments and plots to show how small changes in spread, density, or ignition can completely change the outcome.

NOTE: All functions (besides the starter ones) can be called by functionname(), those all make plots
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation


# STARTER FUNCTIONS - - - - - - - - - - 
# NOTE: first function is the only one used, other 3 were just for testing/visualization help.

# this ones the only one used to generate plots
def forest_fire(isize=10, jsize=10, nstep=8, pspread=0.7, pbare=0.0, pignite=0.0):
    """
    Simulate a stochastic forest-fire spread over a 2D grid.

    Parameters
    ----------
    isize : int
        Number of rows in the forest grid.
    jsize : int
        Number of columns in the forest grid.
    nstep : int
        Number of time steps to simulate.
    pspread : float
        Probability that fire spreads from a burning tree to a neighbor.
    pbare : float
        Probability that a cell starts as bare (no tree).
    pignite : float
        Probability that a tree ignites spontaneously at t=0.

    Returns
    -------
    forest : ndarray
        3D array [time, i, j] storing the state of each cell over time:
        1 = bare/burnt, 2 = tree, 3 = burning.
    """
    # initialize forest array with all trees (state = 2)
    forest = np.zeros((nstep, isize, jsize)) + 2

    # randomly assign bare cells based on pbare
    isbare = np.random.rand(isize, jsize) < pbare
    forest[0][isbare] = 1

    # randomly assign initially burning cells based on pignite
    isburning = np.random.rand(isize, jsize) < pignite
    forest[0][isburning] = 3

    # if no ignition probability, light the center tree manually
    if pignite == 0:
        forest[0, isize // 2, jsize // 2] = 3

    # time evolution loop
    for k in range(nstep - 1):
        # copy current state to next step
        forest[k + 1, :, :] = forest[k, :, :]

        # iterate through all grid cells
        for i in range(isize):
            for j in range(jsize):
                # only spread from burning trees
                if forest[k, i, j] != 3:
                    continue

                # spread fire probabilistically in 4 directions
                if (i > 0) and (pspread > rand()) and (forest[k, i - 1, j] == 2):
                    forest[k + 1, i - 1, j] = 3
                if (i < isize - 1) and (pspread > rand()) and (forest[k, i + 1, j] == 2):
                    forest[k + 1, i + 1, j] = 3
                if (j > 0) and (pspread > rand()) and (forest[k, i, j - 1] == 2):
                    forest[k + 1, i, j - 1] = 3
                if (j < jsize - 1) and (pspread > rand()) and (forest[k, i, j + 1] == 2):
                    forest[k + 1, i, j + 1] = 3

                # burning tree becomes bare after it burns
                forest[k + 1, i, j] = 1

    return forest


def plot_progression(forest):
    """
    Plot how the percentage of forested and bare areas change over time. Not used anywhere, just helped a little.

    Parameters
    ----------
    forest : ndarray
        3D array [time, i, j] of forest states from forest_fire().

    
    """
    ksize, isize, jsize = forest.shape
    npoints = isize * jsize

    # calculate percentage of trees (alive) and bare areas
    loc = forest == 2
    forested = 100 * loc.sum(axis=(1, 2)) / npoints

    loc = forest == 1
    bare = 100 * loc.sum(axis=(1, 2)) / npoints

    # plot both over time
    plt.plot(forested, label='Forested (alive)')
    plt.plot(bare, label='Bare/Burnt')
    plt.xlabel('Time step')
    plt.ylabel('Percent of forest')
    plt.title('Forest Fire Progression')
    plt.legend()
  


def plot_forest2d(forest, step=0):
    """
    Visualize the forest layout at a given time step. Not used anywhere, just helped a little.

    Parameters
    ----------
    forest : ndarray
        3D array [time, i, j] of forest states from forest_fire().
    step : int
        Time step index to visualize.

    
    """
    # define discrete color map: 1=bare, 2=tree, 3=burning
    forest_cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])

    # display selected time step
    plt.imshow(forest[step, :, :], cmap=forest_cmap, vmin=1, vmax=3)
    plt.title(f"Forest at Step {step}")

    # colorbar with state labels
    cbar = plt.colorbar(label='State')
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])
   


def animate_forest(forest):
    """
    Animate the forest-fire evolution over all time steps. Also not used anywhere, just helped a little to visualize.

    Parameters
    ----------
    forest : ndarray
        3D array [time, i, j] of forest states from forest_fire().

    makes a fun little animation
    """
    # create figure and axis
    fig, ax = plt.subplots()

    # same color map as static plot
    cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])
    im = ax.imshow(forest[0], cmap=cmap, vmin=1, vmax=3)
    ax.set_title("Forest Fire Simulation")

    # add colorbar with state labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])

    # update function for animation frames
    def update(frame):
        im.set_data(forest[frame])
        ax.set_title(f"Forest Fire Simulation - Step {frame}")
        return [im]

    # animate through time steps
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=forest.shape[0],
        interval=500,  # milliseconds between frames
        repeat=False
    )




# - - - - - - - 
# TEST RUN SECTION
'''
forest = forest_fire(isize=20, jsize=20, nstep=10, pspread=0.6, pbare=0.2, pignite=0.05)
plot_forest2d(forest, step=3)
animate_forest(forest)
'''
# - - - - - - - - -







## VALIDATION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_forest_3x3_demo():
    """
    Display a 3×3 forest-fire simulation over three time steps (0–2).

    This small-scale visualization shows how the fire propagates cell-by-cell
    when the spread probability is 1, just shows the function works.

        Displays a 1×3 subplot figure with color-coded forest states.
    """
    # run a 3×3 simulation for 4 time steps
    forest = forest_fire(isize=3, jsize=3, nstep=4, pspread=1.0, pbare=0.0, pignite=0.0)

    # define color map: 1=bare/burnt, 2=tree, 3=burning
    forest_cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])

    # create three subplots, one for each time step (0, 1, 2)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    steps = [0, 1, 2]

    # loop over each subplot and draw forest state
    for ax, step in zip(axes, steps):
        im = ax.imshow(forest[step], cmap=forest_cmap, vmin=1, vmax=3)
        ax.set_title(f"Step {step}")
        ax.set_xticks([])
        ax.set_yticks([])

    # shared figure title
    fig.suptitle("3×3 Forest Fire Test (pspread = 1.0)", fontsize=12, y=0.98)

    # add colorbar below all plots for state legend
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.25)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])
    cbar.set_label("Cell State", fontsize=10)

    # adjust layout to leave room for colorbar
    plt.subplots_adjust(bottom=0.25, top=0.85, wspace=0.25)
    plt.show()



def plot_forest_10x20_final():
    """
    Visualize the final burn pattern for a 10×20 forest simulation.

    This one displays the forest’s final state when the spread probability
    is 1, makes a complete burn pattern visualization, again just shows the fire spreads right.

        Displays a single heatmap-style plot with forest states at the final step.
    """
    # run a 10×20 forest fire simulation for 8 steps
    forest = forest_fire(isize=10, jsize=20, nstep=8, pspread=1.0, pbare=0.0, pignite=0.0)

    # color map for states: tan=bare, green=tree, red=burning
    forest_cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])

    # plot final forest state
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(forest[-1], cmap=forest_cmap, vmin=1, vmax=3)

    # title and formatting
    ax.set_title("Final Burn Pattern – 10×20 Grid (pspread = 1.0)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # add horizontal colorbar showing legend for forest states
    cbar = fig.colorbar(im, orientation='horizontal', pad=0.15, fraction=0.05)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])
    cbar.set_label("Cell State", fontsize=10)

    # adjust figure layout for aesthetics
    plt.subplots_adjust(bottom=0.25, top=0.9)
    plt.show()








## 20x20 SUBPLOTS (3, one for each parameter) - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_forest_spread_effect():
    """
    Makes a 2×2 subplot showing the impact of spread probability (pspread)
    on a 20×20 forest fire simulation.

    Two values of pspread (0.3 and 0.7) compared, each shows the initial
    and final forest states to visualize how fire "propagation" strength changes.

        Displays a 2×2 subplot figure comparing different spread probabilities.
    """
    # define color map for cell states
    forest_cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])

    # simulation parameters
    pvals = [0.3, 0.7]   # two spread probabilities
    pbare = 0.1
    pignite = 0.02
    nstep = 25

    # create subplot grid (2 rows × 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    # loop over both spread probabilities
    for i, pspread in enumerate(pvals):
        forest = forest_fire(isize=20, jsize=20, nstep=nstep,
                             pspread=pspread, pbare=pbare, pignite=pignite)

        # left: initial forest
        axes[i, 0].imshow(forest[0], cmap=forest_cmap, vmin=1, vmax=3)
        axes[i, 0].set_title(f"Initial – pspread = {pspread}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # right: final forest
        axes[i, 1].imshow(forest[-1], cmap=forest_cmap, vmin=1, vmax=3)
        axes[i, 1].set_title(f"Final – pspread = {pspread}")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    # add shared colorbar for all subplots
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=forest_cmap,
                        norm=plt.Normalize(vmin=1, vmax=3)),
                        ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.1)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])
    cbar.set_label("Cell State", fontsize=10)

    # figure title and layout adjustment
    fig.suptitle("Effect of Spread Probability on Forest Fire Behavior", fontsize=12, y=0.97)
    plt.subplots_adjust(bottom=0.18, top=0.9, wspace=0.1, hspace=0.25)
    plt.show()



def plot_forest_density_effect():
    """
    Makes a 3×2 subplot showing how forest density (pbare)
    influences fire spread in a 20×20 simulation.

    Three different pbare values (0.1, 0.4, 0.7) representing
    sparse, medium, and dense forests respond to the same spread probability.

        Displays a 3×2 subplot comparing fire spread under different densities (had to represent that "hill" shape, needed 3 here)
    """
    # color map for cell states
    forest_cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])

    # parameters for the test
    pvals = [0.1, 0.4, 0.7]   # below, around, and above critical density
    pspread = 0.7
    pignite = 0.02
    nstep = 35

    # create subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))

    # loop through bare fractions
    for i, pbare in enumerate(pvals):
        forest = forest_fire(isize=20, jsize=20, nstep=nstep,
                             pspread=pspread, pbare=pbare, pignite=pignite)

        # left column: initial configuration
        axes[i, 0].imshow(forest[0], cmap=forest_cmap, vmin=1, vmax=3)
        axes[i, 0].set_title(f"Initial – pbare = {pbare}", pad=8)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # right column: final burned state
        axes[i, 1].imshow(forest[-1], cmap=forest_cmap, vmin=1, vmax=3)
        axes[i, 1].set_title(f"Final – pbare = {pbare}", pad=8)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    # shared colorbar across all subplots
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=forest_cmap,
                        norm=plt.Normalize(vmin=1, vmax=3)),
                        ax=axes, orientation='horizontal',
                        fraction=0.045, pad=0.12)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])
    cbar.set_label("Cell State", fontsize=10, labelpad=6)

    # add title and spacing
    fig.suptitle("Effect of Forest Density on Fire Spread", fontsize=12, y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.18, hspace=0.4)
    plt.show()



def plot_forest_ignite_effect():
    """
    Makes a 2×2 subplot showing how ignition probability (pignite)
    affects the fire spread in the 20×20 grid.

    Used two ignition probabilities (0.02 and 0.1), compared at a fixed
    spread rate and density, illustrates how different ignition rates
    change the burn pattern.

        Displays a 2×2 subplot comparing different ignition probabilities.
    """
    # color map for visualizing states
    forest_cmap = ListedColormap(['tan', 'forestgreen', 'crimson'])

    # parameters for simulation
    pvals = [0.02, 0.1]     # low and high ignition probability
    pspread = 0.7
    pbare = 0.1
    nstep = 25

    # 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    # run for each ignition probability
    for i, pignite in enumerate(pvals):
        forest = forest_fire(isize=20, jsize=20, nstep=nstep,
                             pspread=pspread, pbare=pbare, pignite=pignite)

        # left: initial forest
        axes[i, 0].imshow(forest[0], cmap=forest_cmap, vmin=1, vmax=3)
        axes[i, 0].set_title(f"Initial – pignite = {pignite}", pad=8)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # right: final forest
        axes[i, 1].imshow(forest[-1], cmap=forest_cmap, vmin=1, vmax=3)
        axes[i, 1].set_title(f"Final – pignite = {pignite}", pad=8)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    # shared colorbar for all subplots
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=forest_cmap,
                        norm=plt.Normalize(vmin=1, vmax=3)),
                        ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.12)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Tree', 'Burning'])
    cbar.set_label("Cell State", fontsize=10, labelpad=6)

    # figure title and layout
    fig.suptitle("Effect of Ignition Probability on Fire Spread", fontsize=12, y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.15, hspace=0.4)
    plt.show()





# PROBABILITY SWEEP GRAPHS (2 for pignite, 1 each for pspread and pbare) - - - - - - - - - - - - - - - - - - - - - -  -


def experiment_spread_vs_p(forest_size=20, nstep=12):
    """
    Runs a parameter sweep varying fire spread probability (pspread)
    to see how strongly fire propagation affects overall survival.

    Simulates eleven values of pspread from 0.0 to 1.0 on a 20×20 grid
    with fixed density and ignition rate, plotting the percent of trees
    still alive at the final time step.

        Displays a simple line graph showing how higher pspread leads
        to greater forest loss over time.
    """
    # evenly spaced spread probabilities from 0.0 to 1.0
    pvals = np.linspace(0, 1, 11)
    final_alive = []  # store percent of trees alive at final step

    # main loop over each pspread value
    for p in pvals:
        # run one simulation for this spread probability
        forest = forest_fire(isize=forest_size, jsize=forest_size,
                             nstep=nstep, pspread=p, pbare=0.1, pignite=0.02)

        # find % of cells still containing trees (state = 2)
        alive = np.mean(forest[-1] == 2) * 100
        final_alive.append(alive)

    # plot % alive vs spread probability
    plt.plot(pvals, final_alive, marker='o')
    plt.xlabel("Spread Probability (pspread)")
    plt.ylabel("Final % Forest Remaining")
    plt.title("Impact of Fire Spread Probability on Forest Survival")
    plt.grid(True)
    plt.show()



def experiment_bare_vs_p(forest_size=20, nstep=12):
    """
    Runs a sweep varying the initial bare fraction (pbare)
    to study how forest density changes fire spread behavior.

    Uses eleven density levels from fully dense to mostly empty
    while keeping spread and ignition constant, then measures
    the surviving percent of forest after all steps.

        Displays a line graph showing the threshold-like behavior
        where moderate density burns most effectively.
    """
    # evenly spaced initial bare fractions from 0.0 to 1.0
    pvals = np.linspace(0, 1, 11)
    final_alive = []  # list to store % of forest still alive

    # loop through bare fraction values
    for p in pvals:
        # simulate forest with fixed pspread, varying pbare
        forest = forest_fire(isize=forest_size, jsize=forest_size,
                             nstep=nstep, pspread=0.7, pbare=p, pignite=0.02)

        # compute % trees remaining at the end
        alive = np.mean(forest[-1] == 2) * 100
        final_alive.append(alive)

    # plot results
    plt.plot(pvals, final_alive, marker='o', color='orange')
    plt.xlabel("Initial Bare Fraction (pbare)")
    plt.ylabel("Final % Forest Remaining")
    plt.title("Impact of Initial Forest Density on Fire Spread")
    plt.grid(True)
    plt.show()



def experiment_ignite_vs_p(forest_size=20, nstep=12):
    """
    Runs a sweep varying ignition probability (pignite)
    to test how the number of initial fires affects total burn area.

    Simulates ignition values from 0 to 0.1 while keeping
    spread and density constant, and records how much forest
    remains unburnt by the end.

        Displays a line graph showing how even small increases
        in ignition probability can greatly expand burn coverage.
    """
    # ignition probabilities from 0.0 to 0.1
    pvals = np.linspace(0, 0.1, 11)
    final_alive = []

    # run fire simulations for each ignition level
    for pignite in pvals:
        forest = forest_fire(isize=forest_size, jsize=forest_size,
                             nstep=nstep, pspread=0.7, pbare=0.1, pignite=pignite)

        # compute remaining forest at last frame
        alive = np.mean(forest[-1] == 2) * 100
        final_alive.append(alive)

    # plot final % alive vs ignition probability
    plt.plot(pvals, final_alive, marker='o', color='firebrick')
    plt.xlabel("Ignition Probability (pignite)")
    plt.ylabel("Final % Forest Remaining")
    plt.title("Impact of Ignition Probability on Fire Spread")
    plt.grid(True)
    plt.show()



def experiment_ignite_vs_p_small(forest_size=20, nstep=12):
    """
    Runs a fine-scale ignition probability sweep from 0.00 to 0.02
    to capture sensitivity at very low ignition rates.

    Keeps forest size, spread, and density fixed, focusing only
    on how small ignition differences can change the burn outcome.

        Displays a zoomed-in line plot highlighting early ignition
        thresholds where isolated sparks begin forming larger fires.
    """
    # fine grid of ignition probabilities between 0.00 and 0.02
    pvals = np.linspace(0, 0.02, 11)
    final_alive = []

    # iterate over small ignition probabilities
    for pignite in pvals:
        # simulate at each pignite value (same spread, density)
        forest = forest_fire(isize=forest_size, jsize=forest_size,
                             nstep=nstep, pspread=0.7, pbare=0.1, pignite=pignite)

        # measure % trees that survived
        alive = np.mean(forest[-1] == 2) * 100
        final_alive.append(alive)

    # plot forest survival for small ignition range
    plt.plot(pvals, final_alive, marker='o', color='firebrick')
    plt.xlabel("Ignition Probability (pignite)")
    plt.ylabel("Final % Forest Remaining")
    plt.title("Impact of Ignition Probability on Fire Spread (Zoomed In)")
    plt.grid(True)
    plt.show()






## DISEASE SPREAD MODELING - - - - - - - - - - - - - - - - - - - - - -


def disease_spread(isize=20, jsize=20, nstep=40,
                   pspread=0.6, pimmune=0.2, pinfect=0.02, pfatal=0.2):
    """
    Simulates the spread of a disease in a grid-based population.
    Each cell represents one person that can become infected, recover, or die.

    States:
    0 = immune (cannot be infected)
    1 = healthy (susceptible)
    2 = infected
    3 = dead
    4 = recovered

        Returns a 3D array showing the population state at each time step.
    """
    # initialize grid: everyone starts healthy (state 1)
    grid = np.ones((nstep, isize, jsize))

    # randomly assign immune individuals
    immune_mask = np.random.rand(isize, jsize) < pimmune
    grid[0, immune_mask] = 0

    # infect a small number of initial individuals
    infected_mask = (np.random.rand(isize, jsize) < pinfect) & ~immune_mask
    grid[0, infected_mask] = 2

    # main time loop through simulation steps
    for k in range(nstep - 1):
        grid[k + 1] = grid[k]  # start next frame as copy of current
        for i in range(isize):
            for j in range(jsize):
                if grid[k, i, j] == 2:  # infected person
                    # try to infect 4 neighboring cells (up, down, left, right)
                    for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if 0 <= i + di < isize and 0 <= j + dj < jsize:
                            if grid[k, i + di, j + dj] == 1 and np.random.rand() < pspread:
                                grid[k + 1, i + di, j + dj] = 2
                    # after spreading, decide outcome: death or recovery
                    if np.random.rand() < pfatal:
                        grid[k + 1, i, j] = 3
                    else:
                        grid[k + 1, i, j] = 4
    return grid


def plot_disease_mortality_comparison():
    """
    Visualizes how disease spread differs for two mortality rates (0.1 and 0.6)
    by showing snapshots at several time steps.

        Displays a 2×3 subplot (two mortality rates × three time steps)
        comparing infection and recovery patterns.
    """
    # define colors and labels for each population state
    cmap = ListedColormap(["lightgray", "green", "red", "black", "blue"])
    labels = ['Immune', 'Healthy', 'Infected', 'Dead', 'Recovered']

    # two mortality values to compare and key time steps
    pfatal_values = [0.1, 0.6]
    timesteps = [0, 10, 35]

    # create 2×3 grid of subplots (two pfatal values × three time steps)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    # loop through each mortality case and visualize disease progression
    for row, pfatal in enumerate(pfatal_values):
        disease = disease_spread(nstep=40, pfatal=pfatal)
        for col, step in enumerate(timesteps):
            ax = axes[row, col]
            im = ax.imshow(disease[step], cmap=cmap, vmin=0, vmax=4)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Step {step}")
            if col == 0:
                ax.set_ylabel(f"Pfatal = {pfatal}", fontsize=11)

    # add one shared colorbar across all subplots
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(labels)
    cbar.set_label('Population States')

    # overall figure title
    plt.suptitle("Disease Spread Over Time for Different Mortality Rates", fontsize=13)
    plt.show()



def plot_disease_progression():
    """
    Plots how different population groups (infected, recovered, dead, etc.)
    change over time during one high-mortality simulation.

        Displays a single time-series graph tracking population percentages
        for each state over 25 time steps.
    """
    # run one simulation with moderate infection and high mortality
    nstep = 25
    disease = disease_spread(nstep=nstep, pfatal=0.6, pimmune=0.2, pspread=0.6)

    # calculate % of population in each state per time step
    dead, recovered, infected, healthy, immune = [], [], [], [], []
    total = disease.shape[1] * disease.shape[2]

    for k in range(nstep):
        frame = disease[k]
        dead.append(np.sum(frame == 3) / total * 100)
        recovered.append(np.sum(frame == 4) / total * 100)
        infected.append(np.sum(frame == 2) / total * 100)
        healthy.append(np.sum(frame == 1) / total * 100)
        immune.append(np.sum(frame == 0) / total * 100)

    # plot evolution of each population category over time
    plt.figure(figsize=(8, 5))
    plt.plot(dead, label='Dead', color='black', linewidth=2)
    plt.plot(recovered, label='Recovered', color='blue', linewidth=2)
    plt.plot(infected, label='Infected', color='red', linewidth=2)
    plt.plot(healthy, label='Healthy', color='green', linewidth=2)
    plt.plot(immune, label='Immune (Pre-Vaccinated)', color='gray', linewidth=2)

    plt.xlabel('Time Step')
    plt.ylabel('Population Percentage')
    plt.title('Disease Progression Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
