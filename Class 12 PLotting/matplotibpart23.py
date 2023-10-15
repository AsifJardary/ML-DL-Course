import numpy as np
import matplotlib.pyplot as plt

X=[]
Y=[]
for i in range(1000):
    X.append(i)
    Y.append(np.cos(np.sin(i)))
    plt.plot(X,Y, color="darkorange")
    plt.pause(0.05)
plt.show()
