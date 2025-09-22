'''
turkeys
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

dx = 0.1
x = np.arange(0, 6 * np.pi, dx)
sinx = np.sin(x)
cosx = np.cos(x)

'''
# hard way

fwd_diff = np.zeros(x.size - 1)
for i in range(x.size - 1):
    fwd_diff[i] = x[i+1] - x[i]
'''

# easy way
fwd_diff = (sinx[1:] - sinx[:-1])/dx
back_diff = (sinx[:-1] - sinx[:-2])/dx

plt.plot(x, cosx, label=r'Analytical Derivitve of $\sin{x}$')
plt.plot(x[:-1], fwd_diff, label='Forward Difference')
plt.plot(x[1:], back_diff, label='Backward Difference')
plt.legend(loc='best')
plt.show