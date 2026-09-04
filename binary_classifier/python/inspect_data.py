import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = os.path.abspath(
        os.path.join(os.getcwd(),'..','data')
        )
INPUT_x = os.path.join(DATA_DIR,'x.npy')
INPUT_y = os.path.join(DATA_DIR,'y.npy')
INPUT_c = os.path.join(DATA_DIR,'c.npy')

x = np.load(INPUT_x)
y = np.load(INPUT_y)
c = np.load(INPUT_c)

plt.scatter(x[:, 0], x[:, 1], c=c[y])
plt.show()
