import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 4 * np.pi, 100)
Y_sin = np.sin(X)
Y_cos = np.cos(X)
Y_tan = np.tan(X)

# plt.plot(X, Y_sin)
# plt.plot(X, Y_cos)
plt.plot(X, Y_tan)
plt.grid(True)
plt.show()