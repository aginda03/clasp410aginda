# Final_Proj_410.py
# CLASP 410 Final Project — Solar Panel Tilt + Cloud Simulation

"""

This script simulates how much solar energy a tilted panel can produce over an entire year by modeling the sun’s position, 
panel orientation, and a simple thermal + cloud system. It also sweeps different latitudes and tilt angles to find the 
best setup, then runs Monte Carlo years to see how much the output varies.

To run everthing,
run the file gives latitude tilt optimization and AA monte carlo runs, 
run validation functions at bottom separately
"""


import numpy as np
import matplotlib.pyplot as plt


# ============================================================
## Inputs and basic paramters - - - - - - - - - - - - - - - - - -

# Monthly probability of cloud cover, super simple just to model 
# Ex 0.6 means a 60% chance that a given day in that month is cloudy.
cloud_prob = {
    1: 0.6, 2: 0.55,  3: 0.5, 4: 0.45, 5: 0.35, 6: 0.25,
    7: 0.2, 8: 0.25, 9: 0.35, 10: 0.45,11: 0.55, 12: 0.6
}


# ============================================================
## Basic solar geometry functions - - - - - - - - - - - - - - - - - - -


def solar_declination(day):
    """
    Returns solar declination angle (in degrees) for the given day of year. 
    (where the sun is in the sky throughout the year)
    """
    return 23.44 * np.sin(np.deg2rad((360/365) * (day - 81)))


def hour_angle(hour):
    """
    Hour angle relative to solar noon, 15 deg/hour
    0° at solar noon, negative before noon, positive after.
    Returned in radians.
    """
    return np.deg2rad(15 * (hour - 12))


def solar_position(day, hour, latitude):
    """
    Returns (altitude, azimuth) of the Sun for a given day, hour, and latitude.
    All angles returned in radians.
    
    altitude: angle of the sun above the horizon
    azimuth: direction of the sun (0 = north, 90° = east, 180° = south)
    """
    # Convert latitude to radians
    lat_rad = np.deg2rad(latitude)

    # Solar declination (radians)
    decl = np.deg2rad(solar_declination(day))

    # Hour angle (radians)
    h = hour_angle(hour)

    # Solar altitude angle -----
    # sin(altitude) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(h)
    sin_alt = (
        np.sin(lat_rad)* np.sin(decl) +
        np.cos(lat_rad)*np.cos(decl) *np.cos(h)
    )
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))

    # Solar azimuth angle -----
    # This formula is standard: gives azimuth from north, clockwise
    cos_az = (
        (np.sin(decl) -  np.sin(altitude) * np.sin(lat_rad)) /
        ( np.cos(altitude) * np.cos(lat_rad) + 1e-9)
    )

    # Clip to avoid rounding errors
    cos_az = np.clip(cos_az, -1, 1)

    azimuth = np.arccos(cos_az)

    # Fix quadrant: if hour > solar noon → azimuth should be in western half
    if hour > 12:
        azimuth = 2* np.pi - azimuth

    return altitude, azimuth



def tilted_radiation(altitude, azimuth, tilt_angle):
    """
    Computes clear-sky irradiance on a tilted panel.
    Very simplified model using:
        I = I0 * cos(incident angle)
    with I0 = 1000 W/m^2 (typical clear-sky peak)
    """

    # If sun is below horizon → no radiation
    if altitude <=  0:
        return 0

    I0 = 1000  # W/m^2, clear-sky peak

    # Convert tilt to radians
    tilt = np.deg2rad(tilt_angle)

    # Panel faces south, so panel azimuth = 180° = pi radians
    panel_azimuth = np.pi

    # Compute incidence angle -----
    # Formula: cos(θ) = sin(alt)*cos(tilt) + cos(alt)*sin(tilt)*cos(az - panel_az)
    cos_inc = (
        np.sin(altitude)* np.cos(tilt) +
        np.cos(altitude)*np.sin(tilt) * np.cos(azimuth - panel_azimuth)
    )

    # If sun hits backside → no irradiance
    cos_inc =  max(cos_inc, 0)

    return I0 * cos_inc





# ============================================================
# Clear-Sky Daily Energy Calculation - - - - - -- - - - - - - - - -


def daily_clear_sky_energy(day, latitude, tilt_angle, air_temp=20):
    """
    Computes total electrical energy (Wh/m^2) from a tilted panel
    on a single clear-sky day, gets modified later for clouds,etc.

    Steps (for each hour):
    - find where sun is in the sky (solar_position)
    - compute clear-sky irradiance on the tilted panel (tilted_radiation)
    - estimate panel temperature from an energy balance (panel_temperature)
    - adjust panel efficiency based on temperature (panel_efficiency)
    - convert irradiance -> electrical power and sum over the day
    """
    total_energy = 0.0  # Wh/m^2

    for hour in range(24):
        # Sun position for this hour
        altitude, azimuth = solar_position(day, hour, latitude)

        # Clear-sky irradiance on the tilted panel (W/m^2)
        irradiance = tilted_radiation(altitude, azimuth, tilt_angle)

        # If there is no sun, skip this hour
        if irradiance <= 0:
            continue
        """
                # If irradiance is tiny, skip temperature model and assume panel is air temp
        if irradiance < 5:
            T_panel = air_temp
        else:
            T_panel = panel_temperature(irradiance, air_temp)
            """
        T_panel = panel_temperature(irradiance, air_temp)
        # Temperature-dependent electrical efficiency
        eff = panel_efficiency(T_panel)

        # Electrical power output (W/m^2)
        electrical_power = irradiance  * eff

        # 1-hour time step → Wh/m^2
        total_energy += electrical_power

        

    return total_energy



# ============================================================
# Stochastic cloud Model - - - -  - - - - - - -


def determine_month(day):
    """
    Convert a day-of-year (1–365) into the month number (1–12).
    We do this using fixed month boundaries.
    """
    month_boundaries = [31,  59, 90, 120, 151, 181, 212, 243, 273,304, 334, 365]

    for i, end_day in enumerate(month_boundaries):
        if day <= end_day:
            return i + 1

    return 12  # fallback



def apply_clouds(day, clear_energy):
    """
    Stochastic cloud model:
    - Day is cloudy with month-specific probability
    - Cloudy days have a random reduction factor (not constant)
    """
    month = determine_month(day)

    # Decide if the day is cloudy
    cloudy = np.random.rand() < cloud_prob[month]

    if cloudy:
        # Random reduction (20–50% of clear-sky value)
        reduction = np.random.uniform(0.2, 0.5)
        return clear_energy * reduction
    else:
        return clear_energy


# ============================================================
# Thermal model (losses + efficiency), used above more - - - - - - - - -


def panel_temperature(irradiance, air_temp=20):
    """
    Solve for panel temperature using a steady-state energy balance:
        absorbed solar = convection + conduction + radiation
    We iterate because radiation is nonlinear (T^4).
    """

    # Start with initial guess
    T_panel = air_temp  

    for _ in range(20):  # 20 small iterations is chill
        # Compute losses at current guess
        q_conv =  convection_loss(T_panel, air_temp)
        q_cond = conduction_loss(T_panel, air_temp)
        q_rad  = radiation_loss(T_panel, air_temp)

        q_total = q_conv + q_cond + q_rad  # total heat leaving panel

        # Update temperature guess
        # irradiance [W/m2] goes into heating panel
        # losses [W/m2K] remove energy
        # approximate ΔT step

        # comes from q = k(Tpanel-Tair)
        T_panel = air_temp +  irradiance / ( (q_total / max(T_panel - air_temp, 1e-6)) )

    return T_panel



def convection_loss(T_panel, T_air):
    h = 10  # W/m^2K
    return h * (T_panel - T_air)


def radiation_loss(T_panel, T_air):
    epsilon =  0.9
    sigma = 5.67e-8

    T_p = T_panel + 273.15
    T_sky = (T_air - 20) + 273.15  # rough sky temperature

    return  epsilon * sigma * (T_p**4 - T_sky**4)


def conduction_loss(T_panel, T_air):
    k = 1.0  # W/m^2K
    return k * (T_panel - T_air)


def total_losses(T_panel, T_air=20):
    return (
        convection_loss(T_panel, T_air)
        + radiation_loss(T_panel, T_air)
        + conduction_loss(T_panel, T_air)
    )


def panel_efficiency(T_panel):
    """
    Temperature-dependent panel efficiency.
    """
    eta_ref = 0.20     # 20%
    beta =  0.004       # efficiency drops 0.4% per °C
    return eta_ref * (1 - beta * (T_panel - 25))


# ============================================================
# yearly sim + montecarlo - - - -  - -


def simulate_year(latitude, tilt_angle):
    """
    Simulates one entire year (365 days).
    Each day:
      - compute clear-sky energy
      - randomly apply clouds
    Returns the total annual energy.
    """
    total = 0

    for day in range(1, 366):
        clear_energy = daily_clear_sky_energy(day, latitude, tilt_angle)
        final_energy = apply_clouds(day, clear_energy)
        total += final_energy

    return total


def monte_carlo(latitude, tilt_angle, N=20):
    """
    Run N simulated years and return an array of total annual energy.
    """
    results = np.zeros(N)

    for i in range(N):
        results[i] = simulate_year(latitude, tilt_angle)

    return results




# ============================================================
# Optimal tilt for each latitude + latitude sweep

def optimal_tilt_for_latitude(lat, tilt_min=0, tilt_max=60, tilt_step=5, N=10):
    """
    For a given latitude, sweep tilt angles and return:
        - the tilt angle that produces the highest mean annual energy
        - the corresponding mean annual energy
    """
    tilts = np.arange(tilt_min, tilt_max + tilt_step, tilt_step)
    mean_energies = []

    for tilt in tilts:
        results = monte_carlo(latitude=lat, tilt_angle=tilt, N=N)
        mean_energies.append(np.mean(results))

    mean_energies = np.array(mean_energies)
    best_idx = np.argmax(mean_energies)
    best_tilt = tilts[best_idx]
    best_energy = mean_energies[best_idx]

    return best_tilt, best_energy



def sweep_latitude_with_optimal_tilt(lat_min=0, lat_max=60, lat_step=5, tilt_min=0, tilt_max=60, tilt_step=5, N=10):
    """
    Sweep latitude from lat_min to lat_max. For each latitude:
        - find the optimal tilt angle
        - compute the maximum annual mean energy at that tilt

    Returns:
        lats: array of latitudes
        best_tilts: optimal tilt angle for each latitude
        best_energies: max annual mean energy for each latitude
    """
    lats = np.arange(lat_min, lat_max + lat_step, lat_step)
    best_tilts = []
    best_energies = []

    for lat in lats:
        best_tilt, best_energy = optimal_tilt_for_latitude(
            lat, tilt_min=tilt_min, tilt_max=tilt_max, tilt_step=tilt_step, N=N)
        best_tilts.append(best_tilt)
        best_energies.append(best_energy)

    return lats, np.array(best_tilts), np.array(best_energies)



# Just plotting them, used in main below
def plot_optimal_tilt_vs_latitude(lats, best_tilts):
    plt.figure(figsize=(7,5))
    plt.plot(lats, best_tilts, marker='o')
    plt.xlabel("Latitude (°)")
    plt.ylabel("Optimal Tilt (°)")
    plt.title("Optimal Tilt Angle vs Latitude")
    plt.grid(True)
    plt.show()


def plot_energy_vs_latitude(lats, best_energies):
    plt.figure(figsize=(7,5))
    plt.plot(lats, best_energies, marker='o')
    plt.xlabel("Latitude (°)")
    plt.ylabel("Max Annual Energy (Wh/m²/yr)")
    plt.title("Maximum Solar Energy vs Latitude (each at optimal tilt)")
    plt.grid(True)
    plt.show()






# ============================================================
# main (usually dont do mains just made it easier for now)- - -  - -



if __name__ == "__main__":
    # ========================================================
    # sweep latitude and find optimal tilt for each lat, NOTE: this takes a few minutes to run

    lats, best_tilts, best_energies = sweep_latitude_with_optimal_tilt(
        lat_min=0, lat_max=60, lat_step=10, tilt_min=0, tilt_max=60, tilt_step=5, N=5
    )

    print("Latitude sweep complete!")
    for lat, tilt, energy in zip(lats, best_tilts, best_energies):
        print(f"Latitude {lat}° → Optimal Tilt = {tilt}°, Max Energy = {energy:.1f} Wh/m²/yr")

    # Plot optimal tilt vs latitude
    plot_optimal_tilt_vs_latitude(lats, best_tilts)

    # Plot max energy vs latitude
    plot_energy_vs_latitude(lats, best_energies)



# ========================================================
# Pick Ann Arbor latitude and compute its optimal tilt, NOTE: also takes a minute or so

selected_lat = 42.28    # Ann Arbor latitude

# Compute optimal tilt specifically for this latitude
selected_tilt, _ = optimal_tilt_for_latitude(lat=selected_lat, tilt_min=0,tilt_max=60,
    tilt_step=5, N=10)

print(f"\nUsing latitude {selected_lat}° and optimal tilt {selected_tilt}° for final Monte Carlo...")

# Run larger Monte Carlo for better histogram smoothness
results = monte_carlo(latitude=selected_lat, tilt_angle=selected_tilt, N=200)

print("\nMonte Carlo Summary:")
print("Mean annual energy:", np.mean(results))
print("STD:", np.std(results))
print("10th percentile:", np.percentile(results, 10))
print("50th percentile:", np.percentile(results, 50))
print("90th percentile:", np.percentile(results, 90))

# Histogram of simulated annual energies
plt.hist(results, bins=30, edgecolor='black')
plt.title(f"Annual Energy Distribution\n(lat={selected_lat}°, tilt={selected_tilt}°)")
plt.xlabel("Wh/m² per year")
plt.ylabel("Count (simulated years)")
plt.show()



# ============================================================
# Validation functions, used in paper before th functions used in main

def validate_clear_vs_cloudy(latitude=40, tilt_angle=30, reduction=0.3, day=172):
    """
    Compare the hourly power output on a clear-sky day vs a cloudy day, plots two lines here
    reduction = fraction of clear sky irradiance on cloudy day (e.g., 0.3 = 30%)
    """

    clear = []
    cloudy = []

    for hour in range(24):
        alt, az = solar_position(day, hour, latitude)
        irr = tilted_radiation(alt, az, tilt_angle)

        # skip night
        if irr <= 0:
            clear.append(0)
            cloudy.append(0)
            continue

        # clear sky panel temp + efficiency
        T_clear = panel_temperature(irr)
        eff_clear = panel_efficiency(T_clear)
        power_clear = irr * eff_clear

        # cloudy day irradiance
        irr_cloudy = irr * reduction
        T_cloudy = panel_temperature(irr_cloudy)
        eff_cloudy = panel_efficiency(T_cloudy)
        power_cloudy = irr_cloudy * eff_cloudy

        clear.append(power_clear)
        cloudy.append(power_cloudy)

    hours = np.arange(24)

    plt.figure(figsize=(8,5))
    plt.plot(hours, clear, label="Clear Sky", linewidth=2)
    plt.plot(hours, cloudy, label=f"Cloudy Day ({int(reduction*100)}%)", linewidth=2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Power Output (W/m²)")
    plt.title("Clear Sky vs Cloudy Day Power Output")
    plt.grid(True)
    plt.legend()
    plt.show()


def validate_panel_temperature():
    """
    Produce a table of panel temperatures for different irradiance levels
    using the thermal model. Just helps verify that temperatures scale
    realistically with sunlight intensity.
    """
    irradiances = [200, 400, 600, 800, 1000]   # W/m²
    air_temp = 25 # °C

    print("\n=== Panel Temperature Validation Table ===")
    print(f"{'Irradiance (W/m²)':>18} | {'Panel Temp (°C)':>16}")
    print("-" * 40)

    for irr in irradiances:
        T = panel_temperature(irr, air_temp=air_temp)
        print(f"{irr:18.0f} | {T:16.2f}")


def validate_efficiency():
    print("\n=== Efficiency Validation ===")
    for T in [20, 35, 50, 65, 80]:
        eff = panel_efficiency(T)
        print(f"T = {T:2d} °C → efficiency = {eff*100:.2f}%")


