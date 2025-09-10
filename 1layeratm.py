#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt


"""
One-Layer Atmosphere Energy Balance Model

Assumptions:
  - Atmosphere fully opaque in the longwave (epsilon = 1), transparent in shortwave.
  - Steady State
  - albedo (alpha) constant.
  - Solar irradiance (solar constant) is S0 [W m^-2].

Energy balance equations (fluxes, W m^-2):
  Surface:      sigma * T_s^4 = (1 - alpha) * S0 / 4  +  sigma * T_a^4
  Atmosphere:   sigma * T_s^4 = 2 * sigma * T_a^4

From the atmosphere balance:  T_a^4 = (1/2) * T_s^4
Substitute into surface balance:
    sigma * T_s^4 = (1 - alpha) * S0 / 4  +  sigma * (1/2) * T_s^4
  => (1/2) * sigma * T_s^4 = (1 - alpha) * S0 / 4
  => T_s^4 = (1 - alpha) * S0 / (2 * sigma)
  => T_s(S0, alpha) = [ (1 - alpha) * S0 / (2 * sigma) ]^(1/4)


 Conclusion:


  The results of this plot show that Solar forcing alone can't be the only driver of global warming. The 
  1 layer model predicts a 0.17K increase in temperature from 1900-2000, while were seeing closer to 0.8.
  Matching the increase with just solar irradiance would be completley unreaslistic, 
  where the needed increase in S0 is labeled as well.
  This suggests that other factors like global warming must be a part of this warming as well. 
"""


# Constant
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W m^-2 K^-4]
alpha = 0.33

def equilibrium_temp_one_layer(S0, alpha):
    """Surface temperature (K) for 1-layer fully opaque atmosphere."""
    return (((1 - alpha) * S0) / (2 * SIGMA)) ** 0.25

def required_S0_for_deltaT(S0_base, alpha, dT):
    """Solar constant needed for desired ΔT"""
    T_base = equilibrium_temp_one_layer(S0_base, alpha)
    T_new = T_base + dT
    return S0_base * (T_new / T_base) ** 4


# Main function


# Data
years = np.array([1900, 1950, 2000])
S0s = np.array([1365.0, 1366.5, 1368.0])  # solar constant [W/m^2]

# Observed anomalies (shift so 1900 = 0), makes it easier to visualize the change over time.
observed_anomalies = np.array([-0.4, 0.0, 0.4])
observed_anomalies -= observed_anomalies[0]  # => [0.0, 0.4, 0.8]

# Model anomalies from actual S0 change
T_model = equilibrium_temp_one_layer(S0s, alpha)
dT_model = T_model - T_model[0]

# Calculate ΔS0 needed for observed warming
target_dT = observed_anomalies[-1]  # ~0.8 K
S0_needed_end = required_S0_for_deltaT(S0_base=S0s[0], alpha=alpha, dT=target_dT)
delta_S0 = S0_needed_end - S0s[0]

# Plot model vs observed
plt.figure(figsize=(7, 4.5))
plt.plot(years, dT_model, marker="o", label="Model (solar-only S0)")
plt.plot(years, observed_anomalies, marker="s", label="Observed anomaly")
plt.axhline(0, color="black", linewidth=0.8)

# Place ΔS0 annotation near top of the observed line (last point)
plt.text(years[-1] - 15, observed_anomalies[-1] + 0.05,
         f"ΔS0 needed ≈ {delta_S0:.2f} W/m²",
         fontsize=10,
         bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (K) vs 1900")
plt.title("S0-driven and observed anomaly (1-layer model)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("solar_only_vs_observed.png", dpi=300)
plt.show()