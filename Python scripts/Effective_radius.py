import pynbody as pnb
import numpy as np

def r_effective(p_halo, Rvir):
    haloid = p_halo['grp'][0]
    r_arr = 10**np.linspace(np.log10(0.5), np.log10(Rvir), 100)
    try:
        f = open(f'mass_bins_{haloid}.txt')
        mass = np.loadtxt(f, delimiter = ' ')
    
    except FileNotFoundError:
        mass = []
        for i in range(100):
            current_region = p_halo[pnb.filt.Sphere(r_arr[i])]
            mass.append(current_region.star['mass'].sum())

        np.savetxt(f'mass_bins_{haloid}.txt', mass, delimiter = ' ')
    
    half_mass = mass[-1]/2
    m_arr = abs(mass - half_mass)
    r = r_arr[np.where(m_arr == min(m_arr))[0]]
    return r