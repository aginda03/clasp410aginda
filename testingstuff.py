# testingstuff.py

# adding this part for testing branch

# random wave plot for testing purposes (from AI)

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Define a simple function
def f(x):
    return np.sin(x)

# Generate x values
x = np.linspace(0, 2 * np.pi, 100)

# Calculate y values
y = f(x)

# Calculate derivative using scipy
y_prime = [derivative(f, xi, dx=1e-6) for xi in x]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="sin(x)", color='blue')
plt.plot(x, y_prime, label="Derivative of sin(x)", color='red', linestyle='--')
plt.title("Test Plot: sin(x) and its derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
