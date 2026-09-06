import matplotlib.pyplot as plt
import numpy as np
import os

loss_path = os.path.abspath(os.path.join(os.getcwd(),'..','data','loss.npy'))
loss = np.load(loss_path)
plt.plot(loss)
plt.show()
