import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])
a = min(X)
N = len(X)
T = np.linspace(0.01, 5, 100, True)
enumerator = np.ones((100, 1)).dot(np.expand_dims(X, 1).transpose()) / a
enumerator = np.power(enumerator, np.repeat(-1 / np.expand_dims(T, 1), N, 1))
P = (enumerator.transpose() / enumerator.sum(axis=1)).transpose()
# print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
