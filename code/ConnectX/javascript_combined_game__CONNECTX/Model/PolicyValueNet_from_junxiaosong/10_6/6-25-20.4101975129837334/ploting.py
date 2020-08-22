import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("entropy.log")
print(a.shape)

plt.figure()
plt.title("entropy")
plt.plot(np.arange(a.shape[0]), a)
plt.show()

