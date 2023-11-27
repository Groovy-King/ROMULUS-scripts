import pynbody as pnb
import pynbody.plot.sph as sph
import os
import numpy as np
import matplotlib.pyplot as plt
from pytreegrav import Potential, PotentialTarget

def Pot_Profile(s, r1 = 0, r2 = 500, nbins = 100):
    G = 6.67*10**(-11) * (3.24*10**(-20)) * (2*10**30) * 10**(-6) # Graviitational constant in appropriate units
    r = np.linspace(r1, r2, nbins)
    pot = np.empty(nbins)
    source_pos = s['pos']
    source_mass = s['mass']
    for i in range(nbins - 1):
        target = s.star[pnb.filt.Annulus(r[i], r[i + 1])]
        target_pos = target['pos']
        potential = G * PotentialTarget(np.array(target_pos), np.array(source_pos), np.array(source_mass), method='tree', parallel=True)
        pot[i] = np.mean(potential)

    return pot, r