import numpy as np
import matplotlib.pyplot as plt

# Ambient temperature
Tamb = 20.0  # °C
k = 0.06     # cooling constant [1/min], adjust as needed

# Time array (minutes)
t = np.linspace(0, 20, 500)  # adjust 20 if you need longer range

# Case 1: starts 90°C, ends 60°C
T0_1 = 90.0
Tend_1 = 60.0
T1 = Tamb + (T0_1 - Tamb) * np.exp(-k * t)
t_end_1 = - (1 / k) * np.log((Tend_1 - Tamb) / (T0_1 - Tamb))
#print(f"Case 1 reaches {Tend_1}°C at {t_end_1:.2f} min")

# Case 2: starts 85°C, ends 60°C
T0_2 = 85.0
Tend_2 = 60.0
T2 = Tamb + (T0_2 - Tamb) * np.exp(-k * t)
t_end_2 = - (1 / k) * np.log((Tend_2 - Tamb) / (T0_2 - Tamb))
#print(f"Case 2 reaches {Tend_2}°C at {t_end_2:.2f} min")

# Case 3: starts 90°C, ends 65°C
T0_3 = 90.0
Tend_3 = 65.0
T3 = Tamb + (T0_3 - Tamb) * np.exp(-k * t)
t_end_3 = - (1 / k) * np.log((Tend_3 - Tamb) / (T0_3 - Tamb))
#print(f"Case 3 reaches {Tend_3}°C at {t_end_3:.2f} min")

# -------- Plotting --------
plt.figure(figsize=(8, 5))
plt.plot(t, T1, label=f"Case 1: {T0_1}°C → {Tend_1}°C")
plt.plot(t, T2, label=f"Case 2: {T0_2}°C → {Tend_2}°C")
plt.plot(t, T3, label=f"Case 3: {T0_3}°C → {Tend_3}°C")

# Mark end points
plt.plot(t_end_1, Tend_1, 'o')
plt.annotate(f"{Tend_1}°C @ {t_end_1:.1f} min", (t_end_1, Tend_1), xytext=(5, 5), textcoords="offset points")

plt.plot(t_end_2, Tend_2, 'o')
plt.annotate(f"{Tend_2}°C @ {t_end_2:.1f} min", (t_end_2, Tend_2), xytext=(5, 5), textcoords="offset points")

plt.plot(t_end_3, Tend_3, 'o')
plt.annotate(f"{Tend_3}°C @ {t_end_3:.1f} min", (t_end_3, Tend_3), xytext=(5, 5), textcoords="offset points")

# Ambient line
plt.axhline(Tamb, color='gray', linestyle='--', linewidth=1)
plt.text(0, Tamb + 0.5, f"Ambient = {Tamb}°C")

plt.xlabel("Time [min]")
plt.ylabel("Temperature [°C]")
plt.title("Coffee Cooling – Newton's Law of Cooling")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
