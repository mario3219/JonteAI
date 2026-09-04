import numpy as np
from sklearn.datasets import make_moons
import os

# Paths
OUTPUT = os.path.abspath(
        os.path.join(os.getcwd(),'..','data')
        )
OUTPUT_x = os.path.join(OUTPUT,'x.npy') 
OUTPUT_y = os.path.join(OUTPUT,'y.npy') 
OUTPUT_c = os.path.join(OUTPUT,'c.npy') 

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

x,y = make_moons(n_samples = 300, noise=0.3, random_state=0)
c = np.array(['r','b'])

np.save(OUTPUT_x,x)
np.save(OUTPUT_y,y)
np.save(OUTPUT_c,c)

print(f"Saved X to: {OUTPUT_x}")
print(f"Saved y to: {OUTPUT_y}")
print(f"Saved c to: {OUTPUT_c}")

print(f"x type: {x.dtype}")
print(f"y type: {y.dtype}")
