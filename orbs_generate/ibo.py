import os, glob
import numpy as np
import scipy.linalg
from pyscf import gto, scf, lo, tools, symm
from TDDMRG_CM.utils import util_logbook, util_qm, util_print, util_atoms, util_general
from TDDMRG_CM.utils import util_print as uprint
from TDDMRG_CM.observables import extract_time
from TDDMRG_CM.orbs_generate import util_orbs, analyze_orbs, local_orbs
from TDDMRG_CM.phys_const import au2fs



EXT1 = '.iaoq'

##########################################################################
def get_IBO(mol=None, oiao=None, mo_ref=None, by_symm=False, align_groups=None,
            logbook=None):
    '''
    oiao:
       The orthogonalized IAO in AO basis.
    mo_ref:
       The orbitals acting as the polarization mold based on which the IBO is constructed. In the
       original IBO formulation, it is the Hartree-Fock canonical orbitals.
    align_groups:
       A tuple of tuples. Each inner tuple consists of 1-base IBO indices to be aligned with 
       mo_ref. This input is most likely only useful in linear molecules having orbitals higher 
       than sigma (e.g. pi, delta, etc) where the lobes of these orbitals may not be aligned with
       any Cartesian axes.
    '''

    if mol is None:
        mol = util_atoms.mole(logbook)
        
    if align_groups is not None:
        assert isinstance(align_groups, (list, tuple)), 'align_groups must be a tuple or ' + \
            'list which in turn contains lists or tuples.'

    ovl = mol.intor('int1e_ovlp')
    mf = scf.RHF(mol).run()
    if mo_ref is None:
        mo_ref = mf.mo_coeff[:,mf.mo_occ>0]
    
    #==== Obtain orthogonalized IAOs ====#
    if oiao is None:
        iao = lo.iao.iao(mol, mf.mo_coeff[:,mf.mo_occ>0])
        #OLD e, v = scipy.linalg.eigh(ovl)
        #OLD x_i = v @ np.diag( np.sqrt(e) ) @ v.T   # X^-1 for symmetric orthogonalization
        #OLD iao = x_i @ iao_
        oiao = lo.vec_lowdin(iao, ovl)
    
    #==== Calculate IBOs in AO basis ====#
    if by_symm:
        ibo = np.zeros(mo_ref.shape)
        refs = np.array( symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_ref) )
        for s in set(refs):
            ibo[:,refs == s] = lo.ibo.ibo(mol, mo_ref[:,refs == s], iaos=oiao)
    else:
        ibo = lo.ibo.ibo(mol, mo_ref, iaos=oiao)

    #==== Alignment ====#
    if align_groups is not None:
        # >> It looks like the function of align_groups overlaps with by_symm. <<

        # The task in this block is to determine mix_coef that satisfies:
        #     I = mo_ref.T @ ovl @ ibo_new
        # where
        #     ibo_new = ibo_old @ mix_coef
        # ibo_new is needed in linear molecules because the lobes of ibo_old are not necessarily aligned along
        # any Cartesian axes. The first equation holds under the assumption that the irrep orderings in mo_ref and
        # ibo_new are the same. Hence, the irrep ordering of ibo_new follows that of mo_ref. The columns of ibo_old
        # must be degenerate, the columns of mo_ref may not be degenerate but better are.
        for ik in align_groups:
            ip = tuple( [ik[j]-1 for j in range(0,len(ik))] )
            print('  -> Aligning IBOs:', ip, '(0-based)')
    
            #== Find the most suitable MOs for mo_ref ==#
            c = mo_ref.T @ ovl @ ibo[:,ip]
            idx = np.argmax(np.abs(c), axis=0)
            print('       Indices of the most suitable reference MOs:', idx, '(0-based)')
            mo_ref = mo_ref[:, idx]
    
            #== Compute the new (aligned) IBOs ==#
            mix_coef = np.linalg.inv(mo_ref.T @ ovl @ ibo[:,ip])
            ibo0 = ibo[:,ip] @ mix_coef
            norms = np.einsum('ij, jk, ki -> i', ibo0.T, ovl, ibo0)  # np.diag(ibo.T @ ovl @ ibo)
            ibo[:,ip] = ibo0 / np.sqrt(norms)

    return ibo
##########################################################################


##########################################################################
def get_IBOC(mol=None, oiao=None, ibo=None, mo_ref=None, loc='IBO', by_symm=False, 
             align_groups=None, ortho_thr=1E-8, proj_thr=1E-10, logbook=None):
    '''
    This function calculates the set of vectors orthogonal to IBOs (calculated by get_IBO)
    that are also spanned by IAOs.

    Inputs
    ------
    oiao:
       The orthogonalized IAO in AO basis.
    ibo:
       IBO in AO basis.
    mo_ref:
       The orbitals acting as the polarization mold based on which the IBO is constructed. In the
       original IBO formulation, it is the Hartree-Fock canonical orbitals.
    loc:
       The localization method, available options are 'IBO' and 'PM'

    Outputs
    -------
    iboc:
       The coefficients of IBO-c in AO basis.
    '''

    if mol is None:
        mol = util_atoms.mole(logbook)

    assert loc == 'IBO' or loc == 'PM'
    ovl = mol.intor('int1e_ovlp')
    if (oiao is None or ibo is None) and (mo_ref is None):
        mf = scf.RHF(mol).run()
        mo_ref = mf.mo_coeff[:,mf.mo_occ>0]

    #==== Obtain orthogonalized IAOs ====#
    if oiao is None:
        iao = lo.iao.iao(mol, mo_ref)
        oiao = lo.vec_lowdin(iao, ovl)
        
    #==== Obtain IBOs ====#
    if ibo is None:
        ibo = get_IBO(mol, oiao, mo_ref, by_symm, align_groups)

    #==== Obtain OIAOs in symm. orthogonalized basis ====#
    e, v = scipy.linalg.eigh(ovl)
    x = v @ np.diag( 1/np.sqrt(e) ) @ v.T   # X for symmetric orthogonalization
    x_i = v @ np.diag( np.sqrt(e) ) @ v.T   # X^-1 for symmetric orthogonalization
    oiao_ = x_i @ oiao
    ibo_ = x_i @ ibo
    n_iboc = oiao_.shape[1] - ibo_.shape[1]

    #==== Project OIAO into the complementary space of the IBOs ====#
    # If ibo_ has definite symmetry (irrep), then Q_proj conserves the symmetry of the
    # input vector.
    Q_proj = np.eye(ibo_.shape[0]) - ibo_ @ ibo_.T
    iboc = Q_proj @ oiao_
    norms = np.linalg.norm(iboc, axis=0)
    iboc = iboc[:, norms > proj_thr]
    # ^^This filtering may not necessary. Linear dependencies will be removed during the svd step below anyway.
    iboc = iboc / norms[norms > proj_thr]

    print('No. of nonzero vectors after projection = ', iboc.shape[1])

    #OLD#==== Orthogonalize IBOC via SVD ====#
    #OLDif by_symm:
    #OLD    o = np.array([])
    #OLD    #iboc_s = np.array( symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, x@iboc) )
    #OLD    iboc_s = np.array(util_qm.get_orb_sym(mol, x@iboc))
    #OLD    n_ortho = 0
    #OLD    for s in set(iboc_s):
    #OLD        U, sv, Vt = np.linalg.svd(iboc[:,iboc_s == s], full_matrices=False)
    #OLD        nsym = len(sv[sv > ortho_thr])
    #OLD        o = U[:,0:nsym] if o.size == 0 else np.hstack((o, U[:,0:nsym]))
    #OLD        n_ortho += nsym
    #OLD    iboc = o[:,0:n_ortho]
    #OLDelse:
    #OLD    U, sv, Vt = np.linalg.svd(iboc, full_matrices=False)
    #OLD    n_ortho = len(sv[sv > ortho_thr])
    #OLD    iboc = U[:,0:n_ortho]
    
    
    #==== Orthogonalize IBOC via SVD ====#
    U, sv, Vt = np.linalg.svd(iboc, full_matrices=False)
    n_ortho = len(sv[sv > ortho_thr])
    iboc = U[:,0:n_ortho]

    print('No. of vectors after SVD (cIBOs) = ', iboc.shape[1])
    
    assert n_ortho == n_iboc, f'The number of retained IBOCs ({n_ortho}) must be ' + \
        'the same as the difference between the number of AOs and the number of ' + \
        f'IBOs ({n_iboc}). Try changing ortho_thr.'    
        
    #==== Express orthogonalized IBOC in AO basis ====#
    if by_symm:
        # As it turns out, orthogonalization through SVD above does not necessarily conserve
        # symmetry. This is especially true in linear molecules with point group set to
        # D2h or C2v where the absence of symmetry is reflected in the lobes of pi
        # orbitals not oriented along any Cartesian axes.
        iboc = symm.symmetrize_space(mol, x @ iboc)
    else:
        iboc = x @ iboc    

    #==== Localize IBOC ====#
    if loc == 'IBO':
        #iboc = lo.ibo.ibo(mol, iboc, iaos=oiao)
        iboc = get_IBO(mol, oiao, iboc, by_symm, align_groups)
    elif loc == 'PM':
        assert by_symm, 'When using PM is localization method, by_symm must be True.'
        outs = local_orbs.localize(mol, iboc, loc_subs=[[i+1 for i in range(iboc.shape[1])]])
        iboc = outs['orbs']

    return iboc
##########################################################################


##########################################################################
def analyze(ibo, oiao, mol=None, iboc=None, print_ibo=False, print_oiao=False,
            print_iboc=False, cube_dir=None, logbook=None):

    if mol is None:
        mol = util_atoms.mole(logbook)
    
    ovl = mol.intor('int1e_ovlp')
    print('Number of electrons in mol object = ', mol.nelectron)
    print('Number of AOs = ', mol.nao)
    if iboc is not None:
        print('Sizes of IAO, IBO, and IBOC = ', oiao.shape[1], ibo.shape[1], iboc.shape[1])
    else:
        print('Sizes of IAO and IBO = ', oiao.shape[1], ibo.shape[1])
    print('Trace of overlap matrix of IBO in AO rep. = %10.6f' % np.trace(ibo.T @ ovl @ ibo))
    if iboc is not None:
        print('Trace of overlap matrix of IBOC in AO rep. = %10.6f' %
              np.trace(iboc.T @ ovl @ iboc))
        print('Frobenius norm of overlap matrix between IBOC and IBO = %10.6f' %
              np.linalg.norm(iboc.T @ ovl @ ibo, ord='fro') )

    #==== Analyze IAOs ====#
    print('')
    uprint.print_section('Analysis of IAOs')
    analyze_orbs.analyze(mol, oiao)
    
    #==== Analyze IBOs ====#
    print('')
    uprint.print_section('Analysis of IBOs')
    analyze_orbs.analyze(mol, ibo)
    
    #==== Analyze IBOC's ====#
    if iboc is not None:
        print('')
        uprint.print_section('Analysis of IBOC\'s')
        analyze_orbs.analyze(mol, iboc)

    #==== Print IAOs, IBOs, and IBOCs into cube files ====#
    if print_oiao or print_ibo:
        assert cube_dir is not None, 'cube_dir is needed when printing IBO or IAO into ' + \
            'cube files.'
        ndigit_oiao = len(str(oiao.shape[1]))
        ndigit_ibo = len(str(ibo.shape[1]))
        if not os.path.isdir(cube_dir):
            os.mkdir(cube_dir)
        for i in range(0, max(oiao.shape[1], ibo.shape[1])):
            if print_oiao and i < oiao.shape[1]:
                cubename = cube_dir + '/iao-' + str(i+1).zfill(ndigit_oiao) + '.cube'
                print(f'{i+1:d}) Printing cube files: ', flush=True)
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, oiao[:,i])
            if print_ibo and i < ibo.shape[1]:
                cubename = cube_dir + '/ibo-' + str(i+1).zfill(ndigit_ibo) + '.cube'
                print(f'{i+1:d}) Printing cube files: ', flush=True)
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, ibo[:,i])
    if iboc is not None and print_iboc:
        assert cube_dir is not None, 'cube_dir is needed when printing IBOC into ' + \
            'cube files.'
        ndigit_iboc = len(str(iboc.shape[1]))
        for i in range(0, iboc.shape[1]):
            cubename = cube_dir + '/iboc-' + str(i+1).zfill(ndigit_iboc) + '.cube'
            print(f'{i+1:d}) Printing cube files: ', flush=True)
            print('     ' + cubename)
            tools.cubegen.orbital(mol, cubename, iboc[:,i])
##########################################################################


##########################################################################
def identify_atom(oiao, nmax=2, mol=None, logbook=None):
    '''
    atom is 1-based.
    '''
    if mol is None:
        mol = util_atoms.mole(logbook)

    assert nmax > 0 and nmax <= mol.nao
        
    niao = oiao.shape[1]
    ovl = mol.intor('int1e_ovlp')
    normfac = 1/np.sqrt(np.diag(ovl))    # Normalization factor since AOs might not be normalized.
    overlap = [np.zeros(nmax)]*niao
    atom = [[None]*nmax]*niao
    label = [[None]*nmax]*niao
    for i in range(0, niao):
        ovl_a = np.abs( (ovl @ oiao[:,i]) / normfac )
        # ovl_a contains the overlap between the i-th IAO and the normalized AOs.
        idsort = np.argsort(-ovl_a)
        overlap[i] = ovl_a[idsort[0:nmax]]
        max_label = [mol.ao_labels(fmt=False)[idsort[j]] for j in range(nmax)]
        atom[i] = [(max_label[j][0]+1) for j in range(nmax)]
        label[i] = [max_label[j][1] for j in range(nmax)]
        
    return atom, label, np.array(overlap)
##########################################################################


##########################################################################
def iao_pcharge(oiao, rdm, mol=None, orb=None, nCore=None, nCAS=None, nelCAS=None,
                norm_rdm=False, rthr=0.5, logbook=None):

    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])

        
    assert len(rdm.shape) == 3

    #==== Some constants ====#
    niao = oiao.shape[1]
    nOcc = nCore + nCAS
    ovl = mol.intor('int1e_ovlp')
    
    #==== Calculate the desired orbitals in occupied orbitals basis ====#
    orb_o = orb.T @ ovl @ oiao

    #==== Normalize the RDM ====#
    rdm_full = np.sum( util_qm.make_full_dm(nCore, rdm), axis=0 )
    if norm_rdm:
        tr = np.trace(rdm_full[nCore:nOcc, nCore:nOcc])
        rdm_full[nCore:nOcc, nCore:nOcc] = rdm_full[nCore:nOcc, nCore:nOcc] * nelCAS / tr

    #==== Compute occupation numbers ====#
    occ_orb = np.einsum('ji,jk,ki -> i', orb_o[0:nOcc,:], rdm_full, orb_o[0:nOcc,:]).real

    #==== Get atom IDs of IAOs ====#
    NMAX = 2
    atom, _, overlap = identify_atom(oiao, nmax=NMAX, mol=mol, logbook=None)
    r = overlap[:,1] / overlap[:,0]
    for i in range(niao):
        if r[i] > rthr and atom[i][0] != atom[i][1]:
            uprint.print_warning\
                (f'The dominant atomic contributions to IAO number {i+1:d} ' +
                 f'consist of different atoms: \n' +
                 f'  1. Atom {atom[i][0]+1:d}, overlap = {overlap[i,0]:.6f},\n' +
                 f'  2. Atom {atom[i][1]+1:d}, overlap = {overlap[i,1]:.6f}.\n' +
                 'If you want to suppress this warning, increase rthr (the ' +
                 'current value is {rthr:.6e}). However, note that the closer ' +
                 'the ratio of the above overlaps to unity, the more ' +
                 'ambiguous the association of the above IAO to a single atom is.')

    #==== Compute partial charges ====#
    atomid = np.array( [atom[i][0] for i in range(niao)] )
    q = np.zeros(mol.natm)
    for ia in range(mol.natm):
        q[ia] = mol.atom_charge(ia) - np.sum(occ_orb[atomid-1 == ia])

    return q
##########################################################################


##########################################################################
def td_iao_pcharge(oiao, mol=None, tdir=None, orb=None, nCore=None, nCAS=None, nelCAS=None, 
                   prefix='iao_pcharge', rthr=0.5, simtime_thr=1E-11, tnorm=True, verbose=2,
                   logbook=None):

    '''
    orb:
      AO coefficients of all orbitals.
    '''
    
    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])
    if tdir is None:
        tdir = logbook['sample_dir']
    outname = prefix + EXT1

    #==== Construct the time array ====#
    tt, _, ntevo, rdm_dir = util_general.extract_tevo(tdir)
    idsort = np.argsort(tt)

    #==== Print column titles ====#
    with open(outname, 'w') as outf:
        outf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
        
        outf.write('#%9s %13s  ' % ('Col #1', 'Col #2'))
        for ia in range(mol.natm):
            outf.write('  %12s' % ('Col #' + str(ia+3)))
        outf.write('    %12s' % ('Col #' + str(mol.natm+3)))
        outf.write('\n')

        outf.write('#%9s %13s  ' % ('No.', 'Time (a.u.)'))
        for ia in range(mol.natm):
            label = mol.atom_symbol(ia) + str(ia+1)
            outf.write('  %12s' % label)
        outf.write('    %12s' % 'Total')
        outf.write('\n')
        
    k = 0
    kk = 0
    for i in idsort:
        if kk > 0:
            assert not (tt[i] < t_last), 'Time points are not properly sorted, this is ' \
                'a bug in the program. Report to the developer. ' + \
                f'Current time point = {tt[i]:13.8f}.'

        #==== Load cation RDM1 ====#
        rdm = np.load(rdm_dir[i] + '/1pdm.npy')
        tr = np.sum( np.trace(rdm, axis1=1, axis2=2) ).real
        if tnorm:
            rdm = rdm * nelCAS / tr
            
        #==== When the time point is different from the previous one ====#
        if (kk > 0 and tt[i]-t_last > simtime_thr) or kk == 0:
            if verbose > 1:
                print('%d) Time point: %.5f fs' % (k, tt[i]))
                print('    RDM1 loaded from ' + rdm_dir[i])
                print('    RDM trace = %.8f' % tr)
            echeck = np.linalg.eigvalsh(np.sum(rdm, axis=0))
            qiao = iao_pcharge(oiao, rdm, mol=mol, orb=orb, nCore=nCore, nCAS=nCAS, 
                               nelCAS=nelCAS, norm_rdm=tnorm, rthr=rthr)

            #==== Print correlation indices ====#
            with open(outname, 'a') as outf:
                #== Print time ==#
                outf.write(' %9d %13.8f  ' % (k, tt[i]))
    
                #== Print partial charges ==#
                for ia in range(mol.natm):
                    outf.write('  %12.6f' % qiao[ia])
                outf.write('    %12.6f' % np.sum(qiao))
                outf.write('\n')
                
            #==== Increment unique time point index ====#
            k += 1

        #==== When the time point is similar to the previous one ====#
        elif kk > 0 and tt[i]-t_last < simtime_thr:
            util_print.print_warning\
                ('The data loaded from \n    ' + rdm_dir[i] + '\nhas a time point almost ' +
                 'identical to the previous time point. Duplicate time point = ' +
                 f'{tt[i]:13.8f}')
            echeck_tsim = np.linalg.eigvalsh(np.sum(rdm, axis=0))
            max_error = max(np.abs(echeck_tsim - echeck))
            if max_error > 1E-6:
                util_print.print_warning\
                    (f'The 1RDM loaded at the identical time point {tt[i]:13.8f} yields ' +
                     f'eigenvalues different by up to {max_error:.6e} as the other identical \n' +
                     'time point. Ideally you don\'t want to have such inconcsistency ' +
                     'in your data. Proceed at your own risk.')
            else:
                print('   Data at this identical time point is consistent with the previous ' + \
                      'time point. This is good.\n')

        t_last = tt[i]

        #==== Increment general (non-unique) time point index ====#
        kk += 1
##########################################################################
