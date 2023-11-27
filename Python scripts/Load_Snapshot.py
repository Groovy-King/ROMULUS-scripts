import pynbody as pnb
import os

def load_snapshot0(sim_name, z = -1): #load z=0 snapshot
    if sim_name == 'Romulus25':
        simdir = '/project/rrg-babul-ad/rennehan/romulus_data/cosmo25/'
        param_file = simdir+'cosmo25p.768sg1bwK1BHe75_v4.param'

        ## load snapshot numbers
        filelist = os.listdir(simdir)
        result = [i[-6:] for i in filelist if i.startswith('cosmo25p.') and len(i)==31]
        result.sort()
        result = result[4:]
        snap = result[z] # snapshot id of z=0 snapshot (i.e. the most recent snapshot)
        data_file = simdir+'cosmo25p.768sg1bwK1BHe75.'+snap
        
    elif sim_name == 'RomulusC':
        simdir = '/project/rrg-babul-ad/rennehan/romulus_data/h1.cosmo50/'
        param_file = '/project/rrg-babul-ad/rennehan/romulus_data/h1.cosmo50/h1.cosmo50PLK.1536gst1bwK1BH.param'

        filelist = os.listdir(simdir)
        result = [i[-6:] for i in filelist if i.startswith('h1.cosmo50') and len(i)==35]
        result.sort()
        snap = result[-1]
        data_file = simdir+'h1.cosmo50PLK.1536gst1bwK1BH.'+snap

    elif sim_name == 'RomulusG1':
        simdir = '/project/rrg-babul-ad/rennehan/romulus_data/h102054gs/'
        param_file = '/project/rrg-babul-ad/rennehan/romulus_data/h102054gs/h102054gs.param'

        filelist = os.listdir(simdir)
        result = [i[10:-10] for i in filelist if i.startswith('h102054gs.') and len(i)==26]
        result.sort()
        snap = result[-1]
        data_file = simdir+'h102054gs.'+snap
        
    elif sim_name == 'RomulusG2':
        simdir = '/project/rrg-babul-ad/rennehan/romulus_data/h82651gs/'
        param_file = '/project/rrg-babul-ad/rennehan/romulus_data/h82651gs/h82651gs.param'
        result = [i[9:-10] for i in filelist if i.endswith('amiga.grp') and len(i)==25]
        result.sort()
        snap = result[-1]
        data_file = simdir+'h82651gs.'+snap

    s = pnb.load(data_file, paramfile = param_file)
    s.physical_units()
    return(s)