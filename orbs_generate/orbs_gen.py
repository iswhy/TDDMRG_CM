import numpy as np
from functools import reduce
from pyscf import gto, scf, dft, ao2mo, symm, mcscf
try:
    from pyscf.dmrgscf import DMRGCI
    dmrg_orb = True
except ImportError:
    dmrg_orb = False
from TDDMRG_CM.utils.util_print import print_matrix
from util_orbs import sort_orbs


##########################################################################
def get_rhf_orbs(mol, save_rdm=True, conv_tol=1.0E-7, natorb=False, init_orb=None): 
    '''
    Calculates canonical Hartree-Fock orbitals. For closed-shell systems, RHF calculation 
    is performed. For open-shell systems, ROHF calculation is performed. UHF for open-shell
    systems is not yet supported.
   
    Input parameters:
    ----------------

    mol = Mole object.

    Outputs:
    -------
 
    The output is a dictionary.
    '''

    if natorb:
        print('>>> ATTENTION <<<')
        print('Natural orbitals are specified for get_rhf_orbs which does not have any ' + \
              'effect because in Hartree-Fock method, the natural orbitals are the ' + \
              'same as the canonical orbitals.')
    
    #==== Run HF ====#
    print('\n\n')
    print('==================================')
    print('>>>> HARTREE-FOCK CALCULATION <<<<')
    print('==================================')
    print('')
    print('No. of MO / no. of electrons = %d / (%d, %d)' % 
          (mol.nao, mol.nelec[0], mol.nelec[1]))
    mf = scf.RHF(mol)
    if init_orb is not None:
        mf.init_guess = mf.make_rdm1(mo_coeff=init_orb)
    mf.conv_tol = conv_tol    #1e-7
    mf.kernel()
    orbs, occs, ergs = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    ssq, mult = mf.spin_square()
    print('Spin square = %-10.6f' % ssq)
    print('Spin multiplicity = %-10.6f' % mult)

    orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'erg', 'as')

    
    #==== What to output ====#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    outs['ergs'] = ergs
    if save_rdm: outs['rdm'] = np.diag(occs)
    
    return outs
##########################################################################


##########################################################################
def get_casscf_orbs(mol, nCAS, nelCAS, init_mo, frozen=None, ss=None, ss_shift=None, 
                    twosz=None, wfnsym=None, natorb=False, init_basis=None,
                    state_average=False, sa_weights=None, sort_out=None, save_rdm=True,
                    verbose=2, conv_tol=1.0E-7, canon=True, fcisolver=None, maxM=None,
                    sweep_tol=1.0E-7, dmrg_nthreads=1, set_cas_sym=None, b2_extra=None):
    '''
    Input parameters:
    ----------------

    mol = Mole object of the molecule.
    nCAS = The number of active space orbitals.
    nelCAS = The number of electrons in the active space. The numbers of electrons in CAS
             and in the frozen orbitals (see frozen below) do not have to add up to the total
             number of electrons. The difference will be treated as the core electrons. That
             means there will be ncore/2 orbitals that are doubly occupied throughout the 
             SCF iteration, where ncore is the number of core electrons.
    frozen = Orbitals to be frozen, i.e. not optimized. If given an integer, it is used 
             as the number of the lowest orbitals to be frozen. If it is given a list of
             integers, it must contain the base-1 indices of the orbitals to be frozen.
    init_mo = The guess orbitals to initiate the CASSCF iteration.
    ss = The value of <S^2> = S(S+1) of the desired state. Note that the value of <S^2>
         given to this function is not guaranteed to be achieved at the end of the CASSCF
         procedure. Adjust ss_shift if the final <S^2> is not equal to the value of ss.
    ss_shift = A parameter to control the energy shift of the CASSCF solution when ss is
               not None.
    twosz = The z-projection of the total spin operator (S). Required if state_average is True
            and ss is not None.
    wfnsym = The irreducible representation of the desired multi-electron state.
    natorb = If True, the natural orbitals will be returned, otherwise, the 
             canonical orbitals will be returned.
    save_rdm = Return the 1RDM in the MO rep.
    fcisolver = Controls the use of external FCI solver, at the moment only 'DMRG' is
                is supported. By default, it will use the original CASSCF solver.
    maxM = The maximum bond dimension for the DMRG solver. Only meaningful if fcisolver =
           'DMRG'.

    Outputs:
    -------
 
    The output is a dictionary. With the following elements,
    orbs = The output orbitals. The canonical orbitals if natorb is false, otherwise, the
           natural orbitals.
    occs = The occupation numbers of the output orbitals.
    ergs = If natorb is False, the orbital energies of the output orbitals. If natorb is 
           True, this element is absent.
    rdm = If save_rdm is True, the 1RDM of the solution state in the output orbitals basis.
          Hence, if natorb True, rdm is a diagonal matrix. If save_rdm is False, this 
          element is absent.
    rdm_states = If save_rdm and state_average are both True, the 1RDMs of the solution 
                 states of the state averaging algorithm in the canonical orbitals basis.
                 Note that the basis in which rdm_states is represented is not affected by 
                 what the output orbitals are. That is, unlike rdm whose basis depends on 
                 natorb. If save_rdm is False, this element is absent.
    '''

    
    #==== Set up system (mol) ====#
    assert isinstance(nelCAS, tuple) or isinstance(nelCAS, int)
    ovl = mol.intor('int1e_ovlp')

    #==== Determine the number of core orbitals ====#
    if isinstance(nelCAS, int):
        nelcore = mol.nelectron - nelCAS
    elif isinstance(nelCAS, tuple):
        nelcore = mol.nelectron - (nelCAS[0]+nelCAS[1])
    assert nelcore%2 == 0
    ncore = round(nelcore/2)
    
    #==== Run HF because it is needed by CASSCF ====#
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-7
    mf.kernel()

    #==== Set up the CAS ====#
    print('\n\n')
    print('============================')
    print('>>>> CASSCF CALCULATION <<<<')
    print('============================')
    print('')
    print('No. of MO / no. of electrons = %d / (%d, %d)' % 
          (mol.nao, mol.nelec[0], mol.nelec[1]))
    print('No. of core orbitals / no. of core electrons = %d / %d' %
          (ncore, nelcore))
    if isinstance(nelCAS, int):
        print('No. of CAS orbitals / no. of CAS electrons = %d / %d' %
              (nCAS, nelCAS))
    elif isinstance(nelCAS, tuple):
        print('No. of CAS orbitals / no. of CAS electrons = %d / (%d, %d)' %
              (nCAS, nelCAS[0], nelCAS[1]))
    print('Frozen orbitals = ', end='')
    if isinstance(frozen, int):
        for i in range(0, frozen): print(' %d orbitals' % (i+1), end='')
    elif isinstance(frozen, list):
        for i in frozen: print(' %d' % frozen[i], end='')
    elif frozen is None:
        print('no orbitals frozen', end='')
    print('')
    
    mc = mcscf.CASSCF(mf, ncas=nCAS, nelecas=nelCAS, frozen=frozen)
    mc.conv_tol = conv_tol

    #==== Spin-shifting ====#
    if ss is not None:
        if ss_shift is not None:
            mc.fix_spin_(ss_shift, ss=ss)
        else:
            mc.fix_spin_(ss=ss)

    #==== External FCI solver ====#
    if fcisolver == 'DMRG':
        if not dmrg_orb:
            raise ValueError('get_casscf_orbs: PySCF with the DMRGSCF extension ' +
                             'enabled is required when fcisolver = DMRG.')
        if maxM is None:
            raise ValueError('get_casscf_orbs: maxM is needed when fcisolver = ' + 
                             'DMRG.')
        mc.fcisolver = DMRGCI(mf.mol, maxM=maxM, tol=sweep_tol, num_thrds=dmrg_nthreads,
                              memory = 7)
        mc.internal_rotation = True
        if b2_extra is not None:
            assert isinstance(b2_extra, list)
            print('Block2 extra keyword = ', b2_extra)
            mc.fcisolver.block_extra_keyword = b2_extra
    
        #====   Use the callback function to catch some   ====#
        #==== quantities from inside the iterative solver ====#
        mc.callback = mc.fcisolver.restart_scheduler_()

    #==== Wavefunction symmetry ====#
    if wfnsym is not None:
        mc.fcisolver.wfnsym = wfnsym
    else:
        if mol.groupname == 'c1' or mol.groupname == 'C1':
            mc.fcisolver.wfnsym = 'A'
        else:
            pass
    
    #==== Project initial guess to current basis ====#
    if init_basis is not None:
        print('The guess orbitals are spanned by a different basis, therefore ' +
              'projection of the guess orbitals to the current basis will be ' +
              'performed.')
        mol0 = mol.copy()
        mol0.basis = init_basis
        mol0.build()
        init_mo0 = mcscf.project_init_guess(mc, init_mo, prev_mol=mol0)
    else:
        init_mo0 = init_mo.copy()

    #==== Setting CAS symmetry (if requested) ====#
    if set_cas_sym is not None:
        init_mo0 = mcscf.sort_mo_by_irrep(mc, init_mo0, set_cas_sym)
        # Setting CAS symmetry must come after projection of the initial MO to
        # the desired AO basis representation (if they are in a different AO
        # basis) because the initial MO that is inputto mcscf.sort_mo_by_irrep
        # are assumed to be in the desired AO basis.

    #==== State average ====#
    if state_average:
        assert sa_weights is not None
        nesm = len(sa_weights)
        print(f'State averaging CASSCF will be performed over {nesm} states.')
        mc = mc.state_average_(sa_weights)
        if ss is not None:
            assert twosz is not None
            mc.fcisolver.spin = round(twosz)
            mc.fix_spin_(ss=ss)

    #==== Run CASSCF ====#
    mc.canonicalization = canon
    mc.verbose = 4
    mc.kernel(init_mo0)
    orbs = mc.mo_coeff
    ergs = mc.mo_energy
    if fcisolver is None and not state_average:
        ssq, mult = mcscf.spin_square(mc)
        print('Spin square = %-10.6f' % ssq)
        print('Spin multiplicity = %-10.6f' % mult)

    #==== Determine occupancies and orbital energies ====#
    if state_average:
        rdm_states = np.zeros((mol.nao, mol.nao, nesm))
        for i in range(0,nesm): rdm_states[0:ncore, 0:ncore, i] = 2.0 * np.eye(ncore)
        rdm_states[ncore:ncore+nCAS, ncore:ncore+nCAS, :] = \
            np.stack(mc.fcisolver.states_make_rdm1(mc.ci, nCAS, nelCAS), axis=2)    # 1)
        # 1) mc.fcisolver.states_make_rdm1 produces the states' RDMs in MO (instead of AO)
        #    basis already. Reference:
        #    https://pyscf.org/_modules/pyscf/mcscf/addons.html#StateAverageFCISolver.make_rdm1 .
        rdm_mo = np.dot(rdm_states, sa_weights)     # 2)
        # 2) mc.make_rdm1() can actually be used to obtain the total ensemble RDM in AO
        #    basis which can then be transformed to MO basis to yield the total RDM in MO
        #    basis over all MO, that is, its size is equal to mol.nao x mol.nao, unlike
        #    what mc.fcisolver.states_make_rdm1 produces above, which contains the RDM only
        #    within the active space. The total RMD obtained through use of mc.make_rdm1(),
        #    however, turns out to differ from rdm_mo computed here in the ordering of the
        #    indices. This is bad because the indices of this RDM does not correspond to the
        #    columns of orbs.
    else:
        rdm_ao = reduce(np.dot, (ovl, mc.make_rdm1(), ovl))    # rdm_ao is in AO rep.    3)
        rdm_mo = reduce(np.dot, (orbs.T, rdm_ao, orbs))        # rdm_mo is in MO rep.
        # 3) rdm_ao is in AO rep., needs to transform to an orthogonal rep before feeding
        #    it into symm.eigh below.
    #OLD o_trans = np.hstack(mol.symm_orb)     # ref: https://pyscf.org/pyscf_api_docs/pyscf.symm.html#pyscf.symm.addons.eigh
    

    #==== If natural orbitals are requested, and determine occs ====#
    if natorb:
        print('>>> Computing natural orbitals <<<')
        #== Get natorb in MO rep. ==#
        cas_sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff)   # Probably better use orbs instead of mc.mo_coeff since they are identical anyway.
        natocc, natorb = symm.eigh(rdm_mo, cas_sym)     # 4)
        occs = natocc
        # 4) rdm_mo needs to be in an orthonormal basis rep. to be an input to symm.eigh(),
        #    in this case the MO is chosen as the orthonormal basis.

        #== Transform back from MO rep. to AO rep. ==#
        orbs = orbs @ natorb
        ergs = None

        if sort_out is not None:
            assert sort_out[0] == 'occ', \
                'At the moment sorting using energies when orbital source is ' + \
                'CASSCF is not supported.'
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, sort_out[0], sort_out[1])
        else:
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'occ', 'de')

        #==== Express output RDM in the natural orbitals basis ====#
        rdm_mo = np.diag(occs)
    else:
        occs = np.diag(rdm_mo).copy()
        orbs_ = orbs.copy()
        if sort_out is not None:
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, sort_out[0], sort_out[1])
        else:
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'occ', 'de')
            
        #== Rearrange RDMs to reflect the new ordering of orbs ==#.
        trmat = orbs_.T @ ovl @ orbs
        rdm_mo = trmat.T @ rdm_mo @ trmat
        if state_average:
            for i in range(0,nesm): rdm_states[:,:,i] = trmat.T @ rdm_states[:,:,i] @ trmat


    #==== Analyze ====#
    print('\n')
    print('=====================================')
    print('*** Analysis of the CASSCF result ***')
    print('=====================================')
    print('')
    cc = np.einsum('im, mn, nj -> ij', init_mo0.T, ovl, orbs)
    print('Overlap between the final and initial (guess) orbitals ' +
          '(row -> initial, column -> final):')
    print_matrix(cc)
    print('')
    mc.verbose = verbose
    mc.analyze()
        
        
    #==== What to output ====#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    if ergs is not None: outs['ergs'] = ergs
    if save_rdm: outs['rdm'] = rdm_mo
    if save_rdm and state_average: outs['rdm_states'] = rdm_states

    return outs
##########################################################################


##########################################################################
def get_dft_orbs(mol, xc, conv_tol=1.0E-7, save_rdm=True, natorb=False):
    '''
    Input parameters:
    ----------------

    mol = Mole object.

    Outputs:
    -------
 
    The output is a dictionary.
    '''

    if natorb:
        print('>>> ATTENTION <<<')
        print('Natural orbitals are specified for get_dft_orbs which does not have any ' + \
              'effect because in DFT, the natural orbitals are the same as the canonical ' + \
              'orbitals.')

    #==== Set up system (mol) ====#
    print('\n\n')
    print('=========================')
    print('>>>> DFT CALCULATION <<<<')
    print('=========================')
    print('')
    print('No. of MO / no. of electrons = %d / (%d, %d)' % 
          (mol.nao, mol.nelec[0], mol.nelec[1]))

    #==== Run DFT ====#
    mf = dft.RKS(mol)
    mf.conv_tol = conv_tol
    mf.xc = xc
    mf.kernel()
    orbs, occs, ergs = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    ssq, mult = mf.spin_square()
    print('Spin square = %-10.6f' % ssq)
    print('Spin multiplicity = %-10.6f' % mult)
    orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'erg', 'as')

    #==== What to output ====#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    outs['ergs'] = ergs
    if save_rdm: outs['rdm'] = np.diag(occs)

    return outs
##########################################################################





#=================#
#==== TESTING ====#
#=================#
if __name__ == "__main__":
    #ss = 2
    #shift = None
    #sz = 1
    #
    #mol = gto.M(
    #    atom = 'O 0 0 0; O 0 0 1.2',
    #    basis = 'cc-pvdz',
    #    symmetry = True,
    #    symmetry_subgroup = 'c2v',
    #    spin = 2 * sz)
    #
    #ncore = 2
    #nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
    #nCAS = max(nelCAS) + 2
    
    
    
    
    ss = 0
    shift = None
    sz = 0
    
    mol = gto.M(
        atom = 'C 0 0 -0.6; C 0 0 0.6',
        basis = 'cc-pvdz',
        symmetry = True,
        symmetry_subgroup = 'c2v',
        spin = 2 * sz)
    
    ncore = 2
    nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
    #nCAS = max(nelCAS) + 8
    nCAS = max(nelCAS) + 3
    
    
    
    #ss = 0
    #shift = None
    #sz = 0
    #
    #mol = gto.M(
    #    atom = 'H 0 0 0; H 0 0 1.2',
    #    basis = 'cc-pvdz',
    #    symmetry = 'd2h',
    #    spin = 2 * sz)
    #
    #ncore = 0
    #nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
    #nCAS = max(nelCAS) + 5
    
    
    print('\n\n\n!!! CASSCF !!!')
    orbs = get_casscf_orbs(mol, nCAS, nelCAS, ss=ss, ss_shift=shift,
                          sz=sz, natorb=False, loc_orb=True, loc_type='PM',
                           loc_irrep=True, fcisolver=None)
    
    
    print('\n\n\n!!! CASSCF with DMRG !!!')
    orbs = get_casscf_orbs(mol, nCAS, nelCAS, ss=ss, ss_shift=shift,
                          sz=sz, natorb=False, loc_orb=False, loc_type='PM',
                           loc_irrep=True, fcisolver='DMRG', maxM=400)
    
    
    print('\n\n\n!!! RHF !!!')
    orbs = get_rhf_orbs(mol, sz, loc_orb=False, loc_type='PM', loc_irrep=True)
