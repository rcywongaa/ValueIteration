import matplotlib.pyplot as plt
import numpy as np
import time

a = np.random.random((16, 16))
myplot = plt.imshow(a, cmap='hot', interpolation='nearest')

for i in range(5):
    if i == 0:
        a = np.random.random((16, 16))
        myplot = plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.gcf()
    else:
        a = np.random.random((16, 16))
        myplot.set_data(a)
    plt.pause(1)
