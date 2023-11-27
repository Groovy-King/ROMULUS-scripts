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

G = 6.67*10**(-11) * (3.24*10**(-20)) * (2*10**30) * 10**(-6) # Graviitational constant in appropriate units

s = load_snapshot0('Romulus25')
h = s.halos(ahf_mpi = True) #load halo catalogue

haloid = 52024
h0 = h[haloid]
primary_halo = h0.star[h0.star['grp'] == haloid]

## Determine the center of mass of the given particles, then recenter accordingly
pnb.analysis.halo.center(h0, mode='pot', move_all = True, vel = True)
com = pnb.analysis.halo.center_of_mass(h0)
vcom = pnb.analysis.halo.center_of_mass_velocity(h0)

width = 55
s_filt = primary_halo[pnb.filt.Sphere(width)]

# Finding the virial radius 
prop = s.properties
virovdens = cos.Delta_vir(s)
rdict = {'vir': virovdens,'200': 200, '500': 500, '2500': 2500}
center = com
#dic
MassRadii = cR.get_radius(h0, overdensities=list(rdict.values()), rho_crit=None, prop=prop, precision=1e-2, cen=center, rmax=None)
Rvir = MassRadii[1][virovdens]
r_eff = r_effective(primary_halo, Rvir)

## You can rotate the halo to get a face-on or edge-on view of the central galaxy
pnb.analysis.angmom.sideon(s_filt.star, disksize = '5 kpc', move_all = False, cen = [0,0,0], vcen = [0,0,0])

# Cut out cuboid in edge on view to remove disk
no_disk = h0.star[~pnb.filt.Cuboid(x1 = -width/2, y1 = -3, z1 = -Rvir, x2 = width/2, y2 = 3, z2 = Rvir)]
no_disk = no_disk.star[no_disk.star['grp'] == haloid]

# Use angular momentum to remove disk stars
pos = primary_halo['pos']
vel = primary_halo['vel']
J = np.cross(pos, vel)
J_z = J[:, 2]
J_tot = np.sqrt(np.sum(J**2, axis = 1))

ratio = abs(J_z/J_tot)
ang_cutoff = 0.8
disk = primary_halo[np.where(ratio >= ang_cutoff)[0]]
other = primary_halo[np.where(ratio < ang_cutoff)[0]]

v_disk = np.sqrt(disk['v2'])
v_nodisk = np.sqrt(no_disk['v2'])
v_other = np.sqrt(other['v2'])

# All stars speed distribution
v2 = primary_halo.star['v2']
v = np.sqrt(v2)

q25, q75 = np.percentile(v, [25, 75])
bin_width = 2 * (q75 - q25) * len(v) ** (-1/3)
bins = round((v.max() - v.min()) / bin_width)
print("Freedman-Diaconis number of bins:", bins)

n_bins = int(bins)
hist, bin_edges = np.histogram(v, bins = n_bins)
v_med = []
for i in range(n_bins):
    v_med.append(np.mean(bin_edges[i: i+2]))

v_med = np.array(v_med)
hist = np.array(hist)

opt, cov = curve_fit(Triple_maxwell, v_med, hist, p0 = [100, 10**5, 50, 250, 10**4, 300, 10**4], bounds = [[50, 10**2, 1, 50, 10**3, 100, 10**2], [200, 10**7, 450, 250, 10**6, 400, 10**6]])
sigma_b = opt[0]
coeff = 3 # start the procedure at 3*r_eff
n_max = 10
i = 0
while True:
    r_fid = coeff*r_eff # kpc
    cD = h0[pnb.filt.Sphere(r_fid)]
    mass = cD['mass']
    star_pos = other['pos']
    source_pos = cD['pos']
    potential = PotentialTarget(np.array(star_pos), np.array(source_pos), np.array(mass), method='tree', parallel=True)

    new_pot = G*potential
    kin_energy = v2/2

    total_energy = kin_energy + new_pot
    unbound = other[np.where(total_energy > 0)[0]]
    bound = other[np.where(total_energy <= 0)[0]]

    v_b = np.sqrt(bound['v2'])
    v_ub = np.sqrt(unbound['v2'])

    sigma_unbind = np.std(v_b)
    check = sigma_unbind/sigma_b
    print(i, check)

    if i == n_max:
        break

    elif check > 1.2:
        coeff = coeff*0.9
        i = i + 1
        del cD
        continue

    elif check < 0.8:
        coeff = coeff*1.05
        i = i + 1
        del cD
        continue

    else:
        break

h_bound, bin_edges = np.histogram(v_b, bins = bin_edges)
h_unbound, bin_edges = np.histogram(v_ub, bins = bin_edges)

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
plt.hist(v_b, bins = n_bins, histtype = 'step', color = 'red', label = 'cD (unbind)')
plt.plot(v_med, term2, color = 'blue', linestyle = 'dashed', label = 'Term 2')
plt.hist(v_disk, bins = n_bins, histtype = 'step', color = 'blue', label = 'disk (unbind)')
plt.plot(v_med, term3, color = 'purple', linestyle = 'dashed', label = 'Term 3')
plt.hist(v_ub, bins = n_bins, histtype = 'step', color = 'purple', label = 'IGrL (unbind)')
plt.legend()
plt.grid()
plt.title('3-D Speed Distribution')
plt.xlabel('v (in km/s)')
plt.ylabel('Count')
plt.savefig(f'Halo_{haloid}/3D Speed distributions.png', bbox_inches = 'tight')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
ax1.hist(v, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
ax1.suptitle('All stars')
ax2.hist(v_nodisk, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
ax2.suptitle('Disk removed (Cuboid)')
ax3.hist(v_other, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
ax3.suptitle('Disk removed (Ang. Mom.)')
plt.savefig(f'Halo_{haloid}/Different disk removal techniques.png', bbox_inches = 'tight')

vel_nodisk = no_disk['vel'][:, 2]
vel_other = other['vel'][:, 2]
vel_all = primary_halo['vel'][:, 2]

q25, q75 = np.percentile(v, [25, 75])
bin_width = 2 * (q75 - q25) * len(vel_all) ** (-1/3)
bins = round((vel_all.max() - vel_all.min()) / bin_width)
print("Freedman-Diaconis number of bins:", bins)

n_bins = int(bins)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
ax1.hist(vel_all, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
ax1.suptitle('All stars')
ax2.hist(vel_nodisk, bins = n_bins, histtype = 'step', color = 'black', label = 'Cuboid removed')
ax2.suptitle('Disk removed (Cuboid)')
ax3.hist(vel_other, bins = n_bins, histtype = 'step', color = 'black', label = 'Ang. Mom. removed')
ax3.suptitle('Disk removed (Ang. Mom.)')
plt.savefig(f'Halo_{haloid}/Different disk removal techniques LOS edge_on.png', bbox_inches = 'tight')


## You can rotate the halo to get a face-on or edge-on view of the central galaxy
pnb.analysis.angmom.faceon(s_filt.star, disksize = '5 kpc', move_all = True, cen = [0,0,0], vcen = [0,0,0])
rgb = pnb.plot.stars.render(s_filt.star[np.isnan(s_filt.star['r_lum_den'])==False], width = width, resolution = 1000, mag_range = [17,26],plot = False,ret_im = True)
plt.imshow(rgb[::-1,:],extent=(-width / 2, width / 2, -width / 2, width / 2))
plt.savefig(f'Halo_{haloid}/Face on Image Render.png', bbox_inches = 'tight')

vel_nodisk = no_disk['vel'][:, 2]
vel_other = other['vel'][:, 2]
vel_all = primary_halo['vel'][:, 2]

q25, q75 = np.percentile(v, [25, 75])
bin_width = 2 * (q75 - q25) * len(vel_all) ** (-1/3)
bins = round((vel_all.max() - vel_all.min()) / bin_width)
print("Freedman-Diaconis number of bins:", bins)

n_bins = int(bins)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
ax1.hist(vel_all, bins = n_bins, histtype = 'step', color = 'black', label = 'Romulus data')
ax1.suptitle('All stars')
ax2.hist(vel_nodisk, bins = n_bins, histtype = 'step', color = 'black', label = 'Cuboid removed')
ax2.suptitle('Disk removed (Cuboid)')
ax3.hist(vel_other, bins = n_bins, histtype = 'step', color = 'black', label = 'Ang. Mom. removed')
ax3.suptitle('Disk removed (Ang. Mom.)')
plt.savefig(f'Halo_{haloid}/Different disk removal techniques LOS face_on.png', bbox_inches = 'tight')