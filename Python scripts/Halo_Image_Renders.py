import pynbody as pnb
import pynbody.plot.sph as sph
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import scipy.stats as ss
from pytreegrav import Potential, PotentialTarget
import XIGrM.calculate_R as cR
import XIGrM.cosmology as cos
from scipy.optimize import curve_fit
from Load_Snapshot import load_snapshot0
from decimal import Decimal

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return np.round(float(Decimal(number).scaleb(-fexp(number))), 3)

Redshifts = [0.305,0.356,0.44,0.838,1.061,1.671]
Halo_id = [85619,56009,69110,72806,79292,72729]
BH_masses = [[8.404131e+08, 5.171300e+07],[3.1702528e+07, 1.6211826e+09],[1.1795814e+08, 5.2149562e+08], [3.0548410e+08, 5.6380976e+07],[8.6061864e+07, 3.3600490e+08],[2.8465744e+08, 1.4335941e+08]]

z_data = np.loadtxt('redshifts.txt', delimiter = ' ')
index = z_data[:, 0]
z = z_data[:, 1]

i = 0
idx = int(index[z == Redshifts[i]])
s = load_snapshot0('Romulus25', idx)

h = s.halos(ahf_mpi = True)
haloid = Halo_id[i]

h0 = h[haloid]
## Determine the center of mass of the given particles, then recenter accordingly
pnb.analysis.halo.center(h0, mode='hyb')
## take particles within 35 kpc/h =approx 50kpc sphere
s_filt = h0[pnb.filt.Sphere(55)] 
with open(f'data_high_res_{i}.npy', 'wb') as f:
    width = 55

    ## You can rotate the halo to get a face-on or edge-on view of the central galaxy
    pnb.analysis.angmom.sideon(s_filt.star, disksize = '5 kpc', move_all = False, cen = [0,0,0], vcen = [0,0,0])
    rgb = pnb.plot.stars.render(s_filt.star[np.isnan(s_filt.star['r_lum_den'])==False], width = width, resolution = 1000, mag_range = [17,26],plot = False,ret_im = True)

    pnb.analysis.angmom.faceon(s_filt.star, disksize = '5 kpc', move_all = False, cen = [0,0,0], vcen = [0,0,0])
    rgb2 = pnb.plot.stars.render(s_filt.star[np.isnan(s_filt.star['r_lum_den'])==False], width = width, resolution = 1000, mag_range = [17,26],plot = False,ret_im = True)
    
    np.save(f, rgb)
    np.save(f, rgb2)

