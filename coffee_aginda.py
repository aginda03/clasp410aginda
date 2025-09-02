import numpy as np
import matplotlib.pyplot as plt

# Ambient temperature
Tamb = 20.0  # °C
k = 0.06     # cooling constant [1/min], adjust as needed

# Time array for normal cooling phases
t = np.linspace(0, 10, 500)  # may adjust later

# --- Case 1: normal cooling ---
T0_1 = 90.0
Tend_1 = 60.0
T1 = Tamb + (T0_1 - Tamb) * np.exp(-k * t)
t_end_1 = - (1 / k) * np.log((Tend_1 - Tamb) / (T0_1 - Tamb))
print(f"Case 1 reaches {Tend_1}°C at {t_end_1:.2f} min")

# --- Case 2: instant drop at start (90 → 85), then cool to 60 ---
T0_2_actual = 85.0
Tend_2 = 60.0
T2 = Tamb + (T0_2_actual - Tamb) * np.exp(-k * t)
t_end_2 = - (1 / k) * np.log((Tend_2 - Tamb) / (T0_2_actual - Tamb))
print(f"Case 2 (drop to {T0_2_actual}°C) reaches {Tend_2}°C at {t_end_2:.2f} min")

t_case2 = np.insert(t, 0, 0)
T_case2 = np.insert(T2, 0, 90.0)

# --- Case 3: cool to 65, instant drop to 60 at end ---
T0_3 = 90.0
Tmid_3 = 65.0
Tend_3_final = 60.0

# Time to reach 65 °C
t_mid_3 = - (1 / k) * np.log((Tmid_3 - Tamb) / (T0_3 - Tamb))
print(f"Case 3 cools to {Tmid_3}°C at {t_mid_3:.2f} min, then instant drop to {Tend_3_final}°C")

# Cooling curve only until 65 °C
t3_cooling = np.linspace(0, t_mid_3, 400)
T3_cooling = Tamb + (T0_3 - Tamb) * np.exp(-k * t3_cooling)

# Append instant drop
t_case3 = np.append(t3_cooling, t_mid_3)
T_case3 = np.append(T3_cooling, Tend_3_final)

# --- Plotting ---
plt.figure(figsize=(8, 5))

plt.plot(t, T1, label=f"Case 1: No Cream")
plt.plot(t_case2, T_case2, label=f"Case 2: Cream at Start")
plt.plot(t_case3, T_case3, label=f"Case 3: Cream at End")

# Mark end points
plt.plot(t_end_1, Tend_1, 'o')
plt.annotate(f"{Tend_1}°C @ {t_end_1:.1f} min", (t_end_1, Tend_1), xytext=(5, 5), textcoords="offset points")

plt.plot(t_end_2, Tend_2, 'o')
plt.annotate(f"{Tend_2}°C @ {t_end_2:.1f} min", (t_end_2, Tend_2), xytext=(5, 5), textcoords="offset points")

plt.plot(t_mid_3, Tend_3_final, 'o')
plt.annotate(f"{Tend_3_final}°C @ {t_mid_3:.1f} min", (t_mid_3, Tend_3_final), xytext=(5, 5), textcoords="offset points")

# Ambient line
plt.axhline(Tamb, color='gray', linestyle='--', linewidth=1)
plt.text(0, Tamb + 0.5, f"Ambient = {Tamb}°C")

plt.xlabel("Time [min]")
plt.ylabel("Temperature [°C]")
plt.title("Coffee Cooling – Newton's Law of Cooling (with Instant Drops)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
