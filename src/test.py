import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 1000)
plt.plot(x, np.cos(x), label="cosine")
plt.plot(x, np.sin(x), label="sine")
plt.legend(loc="best")
plt.show()
