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
width = 55

M_c = np.empty(6)
for j in range(6):
    masses = BH_masses[j]
    M_c[j] = (masses[0]*masses[1])**(3/5) / (masses[0] + masses[1])**(1/5)

fig = plt.figure(layout='constrained', figsize=(10, 10/3))
subfigs = fig.subfigures(2, 6, wspace=0.01)
fig.text(-0.015, 0.75, 'Edge-on', va='center', rotation='vertical', fontsize = 10)
fig.text(-0.012, 0.25, 'Face-on', va='center', rotation='vertical', fontsize = 10)
for j in range(6):
    with open(f'data_high_res_{j}.npy', 'rb') as f:
        rgb = np.load(f)
        rgb2 = np.load(f)
    ax1 = subfigs[0, j].subplots(1, 1)
    ax2 = subfigs[1, j].subplots(1, 1)
    ax1.imshow(rgb[::-1,:],extent=(-width / 2, width / 2, -width / 2, width / 2))
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #width = 3*55
    ax2.imshow(rgb2[::-1,:],extent=(-width / 2, width / 2, -width / 2, width / 2))
    ax2.set_xticks([])
    ax2.set_yticks([])
    props = dict(boxstyle='round', facecolor=None, alpha=0.0)
    textstr = '\n'.join((f'z = {Redshifts[j]}', r'M$_{\mathrm{c}} = $' + f'{fman(M_c[j])}' + r'$\times$' + f'10$^{fexp(M_c[j])}$ ' + r'M$_\odot$'))

    # place a text box in upper left in axes coords
    ax2.text(0.05, 0, textstr, transform=ax2.transAxes, fontsize=9, verticalalignment='bottom', bbox=props, color = 'white', horizontalalignment='left')

plt.savefig('Halo Images.png', bbox_inches='tight')