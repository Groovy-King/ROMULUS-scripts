import pynbody as pnb
import pynbody.plot.sph as sph
import os
import numpy as np
from Load_Snapshot import load_snapshot0

s = load_snapshot0('Romulus25')
h = s.halos(ahf_mpi = True)
mass = np.empty(len(h) - 1)

for i in range(1, len(h)):
    h0 = h[i]
    mass[i - 1] = h0.properties['mass']
    del h0
    if i % 1000 == 0:
        print(i)

with open('Halo_Masses.npy', 'wb') as f:
    np.save(f, mass)

sort_idx = np.argsort(mass)
with open('Mass_Order.txt', 'wb') as f:
    np.savetxt(f, sort_idx)