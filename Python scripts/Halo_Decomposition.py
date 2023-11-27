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
from Profile_Functions import *
from Effective_radius import r_effective

s = load_snapshot0('Romulus25')
h = s.halos(ahf_mpi = True)

haloid = 77876
h0 = h[haloid]

## Determine the center of mass of the given particles, then recenter accordingly
pnb.analysis.halo.center(h0, mode='pot', move_all = True, vel = True)
com = pnb.analysis.halo.center_of_mass(h0)
vcom = pnb.analysis.halo.center_of_mass_velocity(h0)
pnb.analysis.angmom.faceon(h0, move_all = False)
#pnb.analysis.angmom.sideon(h0, move_all = False)

#calculate and find the R500 and M500
prop = s.properties
virovdens = cos.Delta_vir(s)
rdict = {'vir': virovdens,'200': 200, '500': 500, '2500': 2500}
center = com
MassRadii = cR.get_radius(h0, overdensities=list(rdict.values()), rho_crit=None, prop=prop, precision=1e-2, cen=center, rmax=None)

# Removing satellite galaxies
primary_halo = h0.star[h0.star['grp'] == haloid]

v2 = primary_halo.star['v2']
v = np.sqrt(v2)

q25, q75 = np.percentile(v, [25, 75])
bin_width = 2 * (q75 - q25) * len(v) ** (-1/3)
n_bins = int(round((v.max() - v.min()) / bin_width))
hist, bin_edges = np.histogram(v, bins = n_bins)
v_med = []
for i in range(n_bins):
    v_med.append(np.mean(bin_edges[i: i+2]))
v_med = np.array(v_med)
hist = np.array(hist)

opt, cov = curve_fit(Triple_maxwell, v_med, hist, p0 = [350, 4000, 350, 80, 10**5, 200, 15000], bounds = ([100, 500, 200, 10, 10**2, 0, 100], [450, 18000, 450, 250, 10**6, 300, np.inf]))
f_v = []
term1 = []
term2 = []
term3 = []
for x in v_med:
    f_v.append(Triple_maxwell(x, *opt))
    term1.append(Maxwell(x, *opt[:2]))
    term2.append(New_maxwell(x, *opt[2:5]))
    term3.append(Maxwell(x, *opt[5:]))

plt.hist(v, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
plt.scatter(v_med, f_v, color = 'green')
plt.plot(v_med, term1, color = 'red', linestyle = 'dashed', label = 'Term 1')
plt.plot(v_med, term2, color = 'blue', linestyle = 'dashed', label = 'Term 2')
plt.plot(v_med, term3, color = 'magenta', linestyle = 'dashed', label = 'Term 3')
plt.legend()
plt.grid()
plt.title('Absolute Velocity Distribution')
plt.xlabel('v (in km/s)')
plt.ylabel('Count')
plt.savefig('Absolute Velocity Distribution.png')

Rvir = MassRadii[1][virovdens]
r_eff = r_effective(primary_halo, Rvir)

r_fid = 3*r_eff # kpc
cD = h0[pnb.filt.Sphere(r_fid)]
mass = cD['mass']
star_pos = primary_halo['pos']
source_pos = cD['pos']
potential = PotentialTarget(np.array(star_pos), np.array(source_pos), np.array(mass), method='tree', parallel=True)

G = 6.67*10**(-11) * (3.24*10**(-20)) * (2*10**30) * 10**(-6)
new_pot = G*potential
kin_energy = v2/2

total_energy = kin_energy + new_pot
unbound = primary_halo[np.where(total_energy > 0)[0]]
bound = primary_halo[np.where(total_energy <= 0)[0]]

v_b = np.sqrt(bound['v2'])
v_ub = np.sqrt(unbound['v2'])

h_bound, bin_edges = np.histogram(v_b, bins = bin_edges)
h_unbound, bin_edges = np.histogram(v_ub, bins = bin_edges)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.hist(v, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
ax1.scatter(v_med, f_v, color = 'green')
ax1.plot(v_med, term1, color = 'red', linestyle = 'dashed', label = 'Term 1')
ax1.plot(v_med, term2, color = 'blue', linestyle = 'dashed', label = 'Term 2')
ax1.plot(v_med, term3, color = 'purple', linestyle = 'dashed', label = 'Term 3')
ax1.legend()
ax1.grid()
ax1.set_xlabel('v (in km/s)')
ax1.set_ylabel('Count')

ax2.plot(v_med, hist, color = 'black', label = 'Romulus data')
ax2.plot(v_med, h_bound, color = 'red', label = 'Term 1')
ax2.plot(v_med, h_unbound, color = 'blue', label = 'Term 2')
ax2.legend()
ax2.grid()
ax2.set_xlabel('v (in km/s)')
plt.savefig('Unbind Velocity profiles.png')