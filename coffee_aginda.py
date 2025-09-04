#!/usr/bin/env python3
"""
Coffee cooling with Newton's law of cooling
- Case 1: normal cooling 90 -> 60
- Case 2: instant drop at start (90 -> 85), then cool to 60
- Case 3: cool from 90 to 65, then instant drop to 60 at that time
"""

import numpy as np
import matplotlib.pyplot as plt

# Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def solve_temp(t, T_init, k, T_amb):
    """
    Temperature at time t using Newton's law of cooling:
    T(t) = T_amb + (T_init - T_amb) * exp(-k t)
    """
    return T_amb + (T_init - T_amb) * np.exp(-k * t)

def solve_time(T_target, T_init, k, T_amb):
    """
    Time needed to go from T_init to T_target under Newton cooling:
    t = -(1/k) * ln((T_target - T_amb)/(T_init - T_amb))
    """
    ratio = (T_target - T_amb) / (T_init - T_amb)
    if ratio <= 0:
        raise ValueError("Invalid temps: T_target must be between T_amb and T_init.")
    return - (1.0 / k) * np.log(ratio)

def verify_script():
    '''
    verify that our implementation is correct
    by checking the time it takes to cool from 110°C to 95°C in 10.76 minutes
    '''
    t_real = 60 * 10.76
    k = np.log(95.0/110./-120.0) / t_real
    t_code = solve_time(120, T_init = 180, T_amb = 70, k=k)

    print("Target Solution is: ", t_real)
    print("Numerical Solution is: ", t_code)
    print("Difference is: ", t_real-t_code)


# Paramters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Tamb = 20.0   # ambient [°C]
k = 0.06      # cooling constant [1/min]

# Defining Each Case - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Case 1: no cream, 90 -> 60 
T0_1, Tend_1 = 90.0, 60.0
t_end_1 = solve_time(Tend_1, T0_1, k=k, T_amb=Tamb)

# Case 2: instant drop (cream) at start (90 -> 85), then 85 -> 60
T0_2_actual, Tend_2 = 85.0, 60.0
t_end_2 = solve_time(Tend_2, T0_2_actual, k=k, T_amb=Tamb)

# Case 3: 90 -> 65 (then instant drop to 60 at that time)
T0_3, Tmid_3, Tend_3 = 90.0, 65.0, 60.0
t_mid_3 = solve_time(Tmid_3, T0_3, k=k, T_amb=Tamb)

# Master time axis sized to include the slowest event
t_max = 1.1 * max(t_end_1, t_end_2, t_mid_3)
t = np.linspace(0, t_max, 600)

# Making each line - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Case 1: normal cooling
T1 = solve_temp(t, T0_1, k=k, T_amb=Tamb)

# Case 2: cool from 85 after an instant drop from 90 at t=0
T2 = solve_temp(t, T0_2_actual, k=k, T_amb=Tamb)
t_case2 = np.insert(t, 0, 0.0)       # prepend t=0
T_case2 = np.insert(T2, 0, 90.0)     # prepend 90°C to show the instant drop

# Case 3: cool until 65, then instant drop to 60 at t_mid_3
t3_cooling = np.linspace(0, t_mid_3, 400)
T3_cooling = solve_temp(t3_cooling, T0_3, k=k, T_amb=Tamb)
t_case3 = np.append(t3_cooling, t_mid_3)          # add drop time again
T_case3 = np.append(T3_cooling, Tend_3)     # instant drop to 60

# Plotting - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plt.figure(figsize=(8, 5))

line1, = plt.plot(t, T1, label="Case 1: No Cream")
line2, = plt.plot(t_case2, T_case2, label="Case 2: Cream at Start")
line3, = plt.plot(t_case3, T_case3, label="Case 3: Cream at End")

# Case 1
plt.plot(t_end_1, Tend_1, 'o', color=line1.get_color())
plt.vlines(t_end_1, Tamb, Tend_1, colors=line1.get_color(), linestyles='dotted')
plt.text(t_end_1, Tamb - 1, f"{t_end_1:.1f} min", ha='center', va='top', color=line1.get_color())

# Case 2
plt.plot(t_end_2, Tend_2, 'o', color=line2.get_color())
plt.vlines(t_end_2, Tamb, Tend_2, colors=line2.get_color(), linestyles='dotted')
plt.text(t_end_2, Tamb - 1, f"{t_end_2:.1f} min", ha='center', va='top', color=line2.get_color())

# Case 3
plt.plot(t_mid_3, Tend_3, 'o', color=line3.get_color())
plt.vlines(t_mid_3, Tamb, Tend_3, colors=line3.get_color(), linestyles='dotted')
plt.text(t_mid_3, Tamb - 1, f"{t_mid_3:.1f} min", ha='center', va='top', color=line3.get_color())

# Ambient line
plt.axhline(Tamb, color='gray', linestyle='--', linewidth=1)
plt.text(0, Tamb + 0.5, f"Ambient = {Tamb}°C")

# Graph Features
plt.xlabel("Time [min]")
plt.ylabel("Temperature [°C]")
plt.title("Coffee Cooling – Newton's Law of Cooling")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

