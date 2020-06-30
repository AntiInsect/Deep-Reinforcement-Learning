import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("loss.log")
print(a.shape)

plt.figure()
plt.title("loss")
plt.plot(np.arange(a.shape[0]), a)
plt.show()

