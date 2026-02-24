import numpy as np
from pyscf import symm
from TDDMRG_CM.utils import util_logbook, util_print, util_atoms, util_general
from TDDMRG_CM.orbs_generate import util_orbs



#######################################################################
#######################################################################
def average(rdm_ref, imtype='ignore', mol=None, tdir=None, orb_a=None, nCore=None,
            nCAS=None, nelCAS=None, logbook=None, simtime_thr=1E-11, tnorm1=True,
            verbose=3):

    '''
    Input parameters
    ----------------
    rdm_ref:
      The active-orbital block of the reference RDM.
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

    if mol is None:
        mol = util_atoms.mole(logbook)
    ovl = mol.intor('int1e_ovlp')
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb_a is None:
        orb_a = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]
    if tdir is None:
        tdir = logbook['sample_dir']
        
    if verbose > 1: 
        print('nCore = ', nCore)
        print('nCAS = ', nCAS)
        print('No. of electrons = ', mol.nelectron)
        print('No. of CAS electrons = ', nelCAS)
        print('No. of AOs = ', mol.nao)
        print('Sz = ', mol.spin/2)
        if logbook is not None:
            print('Orbital path = ', logbook['orb_path'])
        print('Path containing time evolution data = ', tdir)
    
    #==== Get 1RDM path names ====#
    tt, _, nsample, pdm_dir = util_general.extract_tevo(tdir)
    idsort = np.argsort(tt)
    if verbose > 1: print('No. of samples = ', nsample)
    
    #==== Construct the averaged RDM ====#
    # rdm_av is the averaged RDM within the active space only.
    # The averaged RDM in core and virtual spaces are trivial.
    rdm_av = np.zeros((nCAS, nCAS))
    k = 0
    kk = 0
    for i in idsort:
        if kk > 0:
            assert not (tt[i] < t_last), 'Time points are not properly sorted, this is ' \
                'a bug in the program. Report to the developer. ' + \
                f'Current time point = {tt[i]:13.8f}.'
            
        #==== Load cation RDM1 ====#
        rdm1 = np.load(pdm_dir[i] + '/1pdm.npy')
        tr = np.sum( np.trace(rdm1, axis1=1, axis2=2) )
    
        #==== Spin-summed cation RDM1 ====#
        if tnorm1:
            rdm1 = np.sum(rdm1, axis=0) * nelCAS / tr.real
        else:
            rdm1 = np.sum(rdm1, axis=0)
    
        #==== Unique time point ====#
        if (kk > 0 and tt[i]-t_last > simtime_thr) or kk == 0:
            if verbose > 1:
                print('%d) Time point: %.5f fs' % (k, tt[i]))
                print('    Cation RDM1 loaded from ' + pdm_dir[i])
                print('    Trace of the loaded cation 1RDM = ' +
                      '%12.8f (Re), %12.8f (Im)' % (tr.real, tr.imag))
            echeck = np.linalg.eigvalsh(rdm1)
        
            #== Real and imaginary parts analysis ==#
            if verbose > 1:
                rpart = np.sum( np.abs(rdm1.real) )/(nCAS*(nCAS-1))
                ipart = np.sum( np.abs(rdm1.imag) )/(nCAS*(nCAS-1))
                print('    Average of abs. of real components = %.8e' % rpart)
                print('    Average of abs. of imaginary components = %.8e' % ipart)
    
            #== Type of pseudo RDM ==#
            if imtype == 'ignore':
                rdm_av = rdm_av + rdm1.real
            elif imtype == 'abs':
                rdm_av = rdm_av + np.multiply(np.sign(rdm1.real), np.abs(rdm1)) 
            elif imtype == 'sum+':
                rdm_av = rdm_av + (rdm1.real + np.abs(rdm1.imag))
            elif imtype == 'sum-':
                rdm_av = rdm_av + (rdm1.real - np.abs(rdm1.imag))
            else:
                raise ValueError('Unavailable option for imtype.')
    
            #==== Increment unique time point index ====#
            k += 1
            
        #==== Duplicate time point ====#
        elif kk > 0 and tt[i]-t_last < simtime_thr:
            util_print.print_warning\
                ('The data loaded from \n    ' + pdm_dir[i] + '\nhas a time point ' + \
                 'identical to the previous time point. ' + \
                 f'Duplicate time point = {tt[i]:13.8f}')
            echeck_tsim = np.linalg.eigvalsh(rdm1)
            if max(np.abs(echeck_tsim - echeck)) > 1E-6:
                util_print.print_warning\
                    (f'The 1RDM loaded at the identical time point {tt[i]:13.8f} yields ' +
                     'eigenvalues different by more than 1E-6 as the other identical \n' +
                     'time point. Ideally you don\'t want to have such inconcsistency ' +
                     'in your data. Proceed at your own risk.')
            else:
                print('   Data at this identical time point is consistent with the previous ' + \
                      'time point. This is good.\n')
                
        t_last = tt[i]
    
        #==== Increment general (non-unique) time point index ====#
        kk += 1

    #==== Average hole DM ====#
    rdm_av = np.sum(rdm_ref, axis=0) - rdm_av / nsample
        
    #==== Assign irreps to the active orbitals ====#
    sym_a = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orb_a)
    
    #==== Obtain natural orbitals in the active space ====#
    natocc, natorb = symm.eigh(rdm_av, sym_a)
    if verbose > 1:
        print('\nTrace of averaged hole DM = ', np.sum(natocc))
        
    #==== Express the above nat. orbitals in AO basis ====#
    idsort = np.argsort(-np.abs(natocc))   # need the abs value because natocc can be negative.
    orb_av = orb_a @ natorb[:,idsort]
    occ_av = natocc[idsort]
    
    #==== Testing orthogonality ====#
    if verbose > 1:
        itest = orb_av.T @ ovl @ orb_av
        sdiag = np.trace(itest)
        sndiag = np.sum(itest) - sdiag
        print('Orthogonality test of natural orbitals:')
        print('  Trace = ', sdiag)
        print('  Off-diagonal sum = ', sndiag)
    

    return orb_av, occ_av, rdm_av
#######################################################################


#######################################################################
#######################################################################
def correction(borb, orb_av, nav, ortho='svd', ortho_thr=1E-8, corb=None, mol=None,
               nCore=None, logbook=None, verbose=2):

    '''
    Input parameters:
    corb:
      Core orbital.
    borb:
      Base active orbital (no core orbital). Usually, there are nelCAS/2 base
      orbital, where nelCAS is the number of active electrons.
    orb_av:
      Average natural charge orbital (no core orbital).
    nav:
      The first nav average natural charge orbitals will be used to calculate the 
      correction orbital by orthogonal projection onto the core and base orbital
      space. In typical application, assuming the aforementioned projection
      produces no linear dependent orbital, then nav = nCAs - nbase, where nCAS is
      the desired number of active orbital and nbase is the number of base orbital.
    '''

    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if corb is None:   # core orbital
        corb = np.load(logbook['orb_path'])[:,0:nCore]

    if verbose > 1: 
        print('nCore = ', nCore)
        print('No. of electrons = ', mol.nelectron)
        print('No. of AOs = ', mol.nao)
        print('Sz = ', mol.spin/2)
        if logbook is not None:
            print('Orbital path = ', logbook['orb_path'])
            
    ovl = mol.intor('int1e_ovlp')
    
    #==== Base orbitals ====#
    dorb = np.hstack((corb, borb))
    # The dominant orbital consists of the core and base orbitals.
    
    #==== Correction orbitals ====#
    if verbose >= 2: print('Computing correction orbitals:')
    crorb = util_orbs.ortho_project(mol, dorb, orb_av[:,0:nav], ortho, ortho_thr,
                                    verbose)
    
    #==== Orbitals for the excitation space ====#
    if verbose >= 2: print('Computing excitation orbitals:')
    eorb = util_orbs.ortho_project(mol, np.hstack((dorb, crorb)), orb_av, ortho,
                                   ortho_thr, verbose)

    #==== Analysis ====#
    if verbose >= 2:
        print('Number of atomic orbitals = ', mol.nao)
        print('Number of core orbitals = ', corb.shape[1])
        print('Number of base orbitals = ', borb.shape[1])
        print('Number of correction orbitals = ', crorb.shape[1])
        print('Number of excitation orbitals = ', eorb.shape[1])
        print('Frobenius norms analysis (they should be numerically zero):')
        print('  Frobenius norm of overlap matrix between base and correction ' +
              'orbitals = %13.6e' % np.linalg.norm(dorb.T @ ovl @ crorb,
                                                   ord='fro') )
        print('  Frobenius norm of overlap matrix between correction and ' + 
              'excitation orbitals = %13.6e' % np.linalg.norm(crorb.T @ ovl @ eorb,
                                                              ord='fro') )
        print('  Frobenius norm of overlap matrix between base and excitation ' +
              'orbitals = %13.6e' % np.linalg.norm(dorb.T @ ovl @ eorb, ord='fro') )
    
    #==== Output orbitals ====#
    oout = np.hstack((corb, borb, crorb, eorb))
    assert oout.shape[1] == mol.nao, \
        f'nCore = {corb.shape[1]}, nbase = {borb.shape[1]}, ' + \
        f'ncomp = {crorb.shape[1]}, nexc = {eorb.shape[1]}, ' + \
        f'nAO-nCore = {mol.nao-corb.shape[1]}'
    
    #==== Testing orthogonality ====#
    if verbose >= 2:
        itest = oout.T @ ovl @ oout
        sdiag = np.trace(itest)
        sndiag = np.sum(itest) - sdiag
        print('Orthogonality test of the output orbitals:')
        print('  Trace = %-13.8f' % sdiag)
        print('  Off-diagonal sum = %-13.8f' % sndiag)

    return crorb, eorb
#######################################################################
