#!/usr/bin/env python3
# Lab 5 
"""
Lab 5 – Snowball Earth Model

This script runs a super simplified climate model that only looks at latitude.
It uses diffusion, sunlight, radiation, and an ice-albedo flip to show how Earth 
can end up either warm or totally frozen. The whole point is to mess with 
different settings (like diffusivity, emissivity, sunlight, and starting temps) 
and see how the climate reacts.
"""


import numpy as np
import matplotlib.pyplot as plt 

radearth = 6357000.
mxdlyr = 50.
sigma = 5.67e-8
C = 4.2e6
rho = 1020


# -------------------------------
# Required Functions


def gen_grid(npoints=18):
    """
    Makes a basic latitude grid.

    Inputs:
      npoints: how many chunks to split Earth into.

    Outputs:
      dlat: spacing between chunks (degrees)
      lats: center latitude of each chunk

    Just sets up the latitude bands the model uses.
    """
    dlat = 180.0 / npoints
    lats = np.linspace(dlat/2, 180 - dlat/2, npoints)
    return dlat, lats



def temp_warm(lats_in):
    """
    Gives a smooth warm-Earth temp curve.

    Inputs:
      lats_in: latitudes to get temps for.

    Outputs:
      temp: temps at those latitudes (°C)

    Basically takes some reference temps and fits a curve
    so the model has a decent "normal Earth" temperature shape.
    """
    T_warm = np.array([
        -47, -19, -11, 1, 9, 14, 19, 23, 25,
         25, 23, 19, 14, 9, 1, -11, -19, -47
    ])
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    coeffs = np.polyfit(lats, T_warm, 2)
    return coeffs[2] + coeffs[1]*lats_in + coeffs[0]*lats_in**2



def insolation(S0, lats):
    """
    Calculates average yearly sunlight at each latitude.

    Inputs:
      S0: the solar constant (W/m^2)
      lats: array of latitudes where sunlight is calculated

    Outputs:
      ins: yearly-averaged insolation at each latitude (W/m^2)

    Does day/night cycling, seasons, and sun angle stuff to
    figure out how much sunlight each band gets on average.
    """

    # Earth’s tilt - seasons
    max_tilt = 23.5
    # Tiny longitude step so we can average sunlight over a full rotation
    dlong = 0.01

    # Sunlight over a whole day (cos = brightness, negatives = night)
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0  # night side gets zero sunlight

    # Daily averaged sunlight
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Set up array to store sunlight per latitude
    ins = np.zeros(lats.size)

    # Tilt curve for each day of the year (gives seasons)
    tilt = np.array([max_tilt * np.cos(2*np.pi*day/365) for day in range(365)])

    # Loop through each latitude band
    for i, lat in enumerate(lats):
        # Sun angle in the sky for this latitude + day of year
        zen = lat - 90. + tilt
        zen[zen > 90] = 90  # if sun is “below” horizon, call it zero

        # Average sunlight for this latitude through the whole year
        ins[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Final yearly average (one more scaling to match the model)
    return S0_avg * ins / 365.




# ----------------------------------
# Snowball Earth, main model here


def snowball_earth(nlat=18, tfinal=10000,dt=1.0,lam=100.,emiss=1.0,
    init_cond=temp_warm,apply_spherecorr=False,albice=.6,albgnd=.3,
    apply_insol=False,solar=1370):
    """
    Main climate model. Runs the whole Snowball Earth simulation.

    Inputs:
      nlat: number of latitude bands
      tfinal: total run time in years
      dt: timestep in years
      lam: diffusivity (how fast heat spreads)
      emiss: emissivity for IR cooling
      init_cond: starting temperature (function or array)
      apply_spherecorr: turn spherical geometry correction on/off
      albice: albedo for ice
      albgnd: albedo for non-ice
      apply_insol: turn sunlight physics on/off
      solar: solar constant (W/m^2)

    Outputs:
      lats: latitude array (degrees)
      temp: final temperature profile (°C)
      K: diffusion matrix (just for checking stuff)

    Basically: sets up all the physics (diffusion, sunlight, radiation,
    albedo changes), then steps forward in time and returns the final temps.
    """

    # make the latitude grid
    dlat, lats = gen_grid(nlat)
    # convert lat spacing into meters for diffusion math
    dy = np.pi * radearth / nlat

    # area of each latitude band (used in sphere correction)
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180. * lats)

    # first-derivative operator (used for sphere correction)
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1.
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1.
    B[0,:] = B[-1,:] = 0.

    dAxz = np.matmul(B, Axz)

    # number of timesteps to run
    nsteps = int(tfinal / dt)
    # convert dt into seconds so physics has real units
    dt_sec = dt * 365 * 24 * 3600

    temp = np.zeros(nlat)

    # starting temps — either a function or a preset array
    if callable(init_cond):
        temp = init_cond(lats)
    else:
        temp[:] = init_cond

    # diffusion matrix K (basic second derivative setup)
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2.
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1.
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1.
    K[0, 1] = 2.
    K[-1, -2] = 2.

    # scale diffusion by dy^2 so units make sense
    K *= 1 / dy**2

    # build the L matrix and its inverse (used to advance diffusion)
    L = np.eye(nlat) - dt_sec * lam * K
    Linv = np.linalg.inv(L)

    # set up starting albedo (ice or not)
    albedo = np.zeros(nlat)
    loc_ice = temp <= -10
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # sunlight at each latitude
    insol = insolation(solar, lats)

    # -----------------------------------

    for i in range(nsteps):

        # update which bands are frozen (changes albedo)
        loc_ice = temp <= -10
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # spherical geometry correction (optional)
        if apply_spherecorr:
            sphere_corr = (lam*dt_sec) / (4*Axz*dy**2) * np.matmul(B, temp) * dAxz
        else:
            sphere_corr = 0

        # sunlight + IR cooling (optional)
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(temp+273)**4
            temp += dt_sec * radiative / (rho*C*mxdlyr)

        # main diffusion update
        temp = np.matmul(Linv, temp + sphere_corr)

    return lats, temp, K





# -------------------------------
# LAB QUESTIONS


# QUESTION 1 

def model_validation():
    """
    Plots the three runs for Question 1:
      1. diffusion only
      2. diffusion + spherical correction
      3. full model (radiation + real albedo)
    Just checks that the model behaves normally.

    model_validation()
    """

    # 0. grid + warm starting temps
    dlat, lats = gen_grid(18)
    temp_init = temp_warm(lats)

    # 1. diffusion only (constant albedo = 0.3)
    lats1, temp_diff, _ = snowball_earth(
        nlat=18, tfinal=10000, dt=1.0, lam=100, emiss=0,
        apply_spherecorr=False, apply_insol=False,
        albice=0.3, albgnd=0.3, init_cond=temp_warm
    )

    # 2. diffusion + spherical correction (still constant albedo)
    lats2, temp_sphere, _ = snowball_earth(
        nlat=18, tfinal=10000, dt=1.0, lam=100, emiss=0,
        apply_spherecorr=True, apply_insol=False,
        albice=0.3, albgnd=0.3, init_cond=temp_warm
    )

    # 3. full model (diffusion + spherical + radiation with real albedo)
    lats3, temp_full, _ = snowball_earth(
        nlat=18, tfinal=10000, dt=1.0, lam=100, emiss=1.0,
        apply_spherecorr=True, apply_insol=True,
        albice=0.6, albgnd=0.3, init_cond=temp_warm
    )

    # 4. plot everything
    plt.figure(figsize=(8,5))
    plt.plot(lats - 90, temp_init,  label="Initial Condition (warm Earth)")
    plt.plot(lats1 - 90, temp_diff, label="1. Diffusion Only")
    plt.plot(lats2 - 90, temp_sphere, label="2. Diffusion + Spherical Corr.")
    plt.plot(lats3 - 90, temp_full,  label="3. Full Model")

    plt.xlabel("Latitude (deg)")
    plt.ylabel("Temperature (°C)")
    plt.title("Model Validation")
    plt.legend()
    plt.tight_layout()
    plt.show()




# QUESTION 2

def sweep_lambda():
    """
    Sweeps through a bunch of λ (diffusivity) values and plots
    how they change the final temperature profile.
    Basically shows how fast heat-spreading affects the model.

    sweep_lambda()
    """

    # basic grid + warm target curve to compare against
    dlat, lats = gen_grid(18)
    target = temp_warm(lats)

    plt.figure(figsize=(8,5))
    plt.plot(lats - 90, target, label="Warm Earth Target", linewidth=3)

    # try a bunch of λ values and plot each one
    for lam in [0, 25, 50, 75, 100, 150]:
        _, temp_run, _ = snowball_earth(
            nlat=18, tfinal=10000, dt=1.0,
            lam=lam, emiss=1.0,
            apply_spherecorr=True,
            apply_insol=True,
            init_cond=temp_warm,
            albice=0.6, albgnd=0.3
        )
        plt.plot(lats - 90, temp_run, label=f"λ = {lam}")

    plt.xlabel("Latitude (deg)")
    plt.ylabel("Temperature (°C)")
    plt.title("Effect of Diffusivity λ on Equilibrium Temperature")
    plt.legend()
    plt.tight_layout()
    plt.show()


def sweep_emissivity():
    """
    Sweeps through emissivity values and plots how they change
    the temp profile. Higher emissivity = more IR cooling.

    sweep_emissivity()
    """

    # grid + target “warm Earth” curve
    dlat, lats = gen_grid(18)
    target = temp_warm(lats)

    plt.figure(figsize=(8,5))
    plt.plot(lats - 90, target, label="Warm Earth Target", linewidth=3)

    # try a bunch of emissivity values with λ held constant
    for emiss in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        _, temp_run, _ = snowball_earth(
            nlat=18, tfinal=10000, dt=1.0,
            lam=100, emiss=emiss,
            apply_spherecorr=True,
            apply_insol=True,
            init_cond=temp_warm,
            albice=0.6, albgnd=0.3
        )
        plt.plot(lats - 90, temp_run, label=f"ε = {emiss}")

    plt.xlabel("Latitude (deg)")
    plt.ylabel("Temperature (°C)")
    plt.title("Effect of Emissivity ε on Equilibrium Temperature")
    plt.legend()
    plt.tight_layout()
    plt.show()


 # QUESTION 3

def IC_tests():
    """
    Runs the three initial-condition tests for Question 3:
      1. start super hot
      2. start super cold
      3. start warm but force ice everywhere (flash freeze)
    Shows how the model can end up in different equilibrium states.

    IC_tests()
    """

    # basic grid
    dlat, lats = gen_grid(18)

    # 1. HOT EARTH (everything starts at +60°C)
    hot_init = np.full(lats.size, 60.0)
    lats_hot, temp_hot, _ = snowball_earth(
        nlat=18, tfinal=10000, dt=1.0,
        lam=60, emiss=0.7,
        apply_spherecorr=True, apply_insol=True,
        init_cond=hot_init,
        albice=0.6, albgnd=0.3
    )

    # 2. COLD EARTH (everything starts at –60°C)
    cold_init = np.full(lats.size, -60.0)
    lats_cold, temp_cold, _ = snowball_earth(
        nlat=18, tfinal=10000, dt=1.0,
        lam=60, emiss=0.7,
        apply_spherecorr=True, apply_insol=True,
        init_cond=cold_init,
        albice=0.6, albgnd=0.3
    )

    # 3. FLASH-FREEZE (start warm, but force ice albedo instantly)
    warm_init = temp_warm(lats)
    lats_flash, temp_flash, _ = snowball_earth(
        nlat=18, tfinal=10000, dt=1.0,
        lam=60, emiss=0.7,
        apply_spherecorr=True, apply_insol=True,
        init_cond=warm_init,
        albice=0.6, albgnd=0.6   # both set to ice albedo → instantly frozen
    )

    # make the plot
    plt.figure(figsize=(8,5))
    plt.plot(lats - 90, temp_hot,   label="Hot Earth (60°C)")
    plt.plot(lats - 90, temp_cold,  label="Cold Earth (-60°C)")
    plt.plot(lats - 90, temp_flash, label="Flash-Frozen Warm Earth")

    plt.xlabel("Latitude (deg)")
    plt.ylabel("Temperature (°C)")
    plt.title("Initial Condition Effects")
    plt.legend()
    plt.tight_layout()
    plt.show()




# QUESTION 4

def plot_hysteresis():
    """
    Runs the solar-forcing hysteresis test.
    Sweeps γ upward (cold → warm) and then downward (warm → cold)
    to show the climate “memory” and the split hysteresis loop.

    plot_hysteresis()
    """

    dlat, lats = gen_grid(18)

    # gamma ranges for the sweep
    gammas_up = np.arange(0.4, 1.45, 0.05)     # going warmer
    gammas_down = np.arange(1.4, 0.35, -0.05)  # going colder

    # where global mean temps get stored
    temps_up = []
    temps_down = []

  
    # 1. up sweep (start super cold)
  
    temp_init = np.full(lats.size, -60.0)   # frozen Earth start

    for gamma in gammas_up:
        _, temp_final, _ = snowball_earth(
            nlat=18, tfinal=10000, dt=1.0,
            lam=100, emiss=1.0,
            apply_spherecorr=True, apply_insol=True,
            init_cond=temp_init,
            albice=0.6, albgnd=0.3,
            solar=1370 * gamma
        )

        temps_up.append(temp_final.mean())   # global mean temp
        temp_init = temp_final.copy()        # pass final → next run


    
    # 2. down sweep (start from warm)

    temp_init = temp_final.copy()   # start from whatever “warm” we just got

    for gamma in gammas_down:
        _, temp_final, _ = snowball_earth(
            nlat=18, tfinal=10000, dt=1.0,
            lam=100, emiss=1.0,
            apply_spherecorr=True, apply_insol=True,
            init_cond=temp_init,
            albice=0.6, albgnd=0.3,
            solar=1370 * gamma
        )

        temps_down.append(temp_final.mean())
        temp_init = temp_final.copy()


    # Plot that hysteresis curve
    
    plt.figure(figsize=(8,6))
    plt.plot(gammas_up, temps_up,   'o-', label="Cold Start → Increasing γ")
    plt.plot(gammas_down, temps_down,'o-', label="Warm Start → Decreasing γ")

    plt.xlabel("Solar Multiplier γ")
    plt.ylabel("Global Mean Temperature (°C)")
    plt.title("Solar Forcing Hysteresis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
