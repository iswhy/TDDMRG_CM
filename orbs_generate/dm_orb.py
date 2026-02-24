import numpy as np
from pyscf import symm
from TDDMRG_CM.utils import util_logbook, util_print, util_atoms, util_general
from TDDMRG_CM.orbs_generate import util_orbs, hdm_orb



#######################################################################
#######################################################################
def average(imtype='ignore', mol=None, tdir=None, orb_a=None, nCore=None,
            nCAS=None, nelCAS=None, logbook=None, simtime_thr=1E-11, tnorm1=True,
            verbose=3):

    '''
    Input parameters
    ----------------
    orb_a:
      Active orbital.
    nelCAS:
      No. of active electrons during time evolution.

    Return parameters
    -----------------
    orb_av:
      Average natural charge orbital (no core orbital).
    occ_av:
      Average hole occupancies of orb_av (no core occupancies).
    rdm_av:
      Averaged hole DM in MO basis where MO is the orbital used in the time
      evolution.
    '''


    if nCAS is None:
        nCAS = logbook['nCAS']

    rdm_ref = np.zeros((nCAS, nCAS))
    orb_av, occ_av, rdm_av = hdm_orb.average(
        rdm_ref, imtype, mol, tdir, orb_a, nCore, nCAS, nelCAS, logbook, simtime_thr, 
        tnorm1, verbose)
    # occ_av contains negative values sorted from the most negative to the most positive.
    # So, one only needs to flip the sign of occ_av without reordering it and orb_av.
    # One also needs to multiply rdm_av with -1.

    occ_av = -occ_av
    rdm_av = -rdm_av
    

    return orb_av, occ_av, rdm_av
#######################################################################


