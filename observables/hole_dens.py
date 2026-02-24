import sys, os, glob
import numpy as np
from pyscf import symm, tools
from TDDMRG_CM.phys_const import rad2deg, ang2bohr
from TDDMRG_CM.utils import util_atoms, util_print, util_general

EXT1 = '.tpl'
EXT2 = '.tvl.cube'


####################################################
def eval_xyplane(rdm0, z, boundx, boundy, xyorig=(0.0, 0.0), mol=None, tdir=None, 
                 orb=None, nCore=None, nCAS=None, nelCAS=None, tnorm0=False, tnorm1=True, 
                 prefix='plane_hole', print_cart=False, simtime_thr=1E-11, verbose=2,
                 logbook=None):

    assert len(xyorig) == 2
    orig = (xyorig[0], xyorig[1], 0.0)
    eval_plane(rdm0, np.array((0.0, 0.0, 1.0)), z, orig, boundx, boundy, 0.0, mol, tdir, orb, nCore,
               nCAS, nelCAS, tnorm0, tnorm1, prefix, print_cart, simtime_thr, verbose,
               logbook)
####################################################


####################################################
def eval_xzplane(rdm0, y, boundx, boundz, xzorig=(0.0, 0.0), mol=None, tdir=None, 
                 orb=None, nCore=None, nCAS=None, nelCAS=None, tnorm0=False, tnorm1=True, 
                 prefix='plane_hole', print_cart=False, simtime_thr=1E-11, verbose=2,
                 logbook=None):

    assert len(xzorig) == 2
    orig = (xzorig[0], 0.0, xzorig[1])
    boundz0 = (-boundz[2], boundz[1], -boundz[0])
    eval_plane(rdm0, (0.0, 1.0, 0.0), y, orig, boundz0, boundx, 0.0, mol, tdir, orb, 
               nCore, nCAS, nelCAS, tnorm0, tnorm1, prefix, print_cart, simtime_thr, 
               verbose, logbook)
####################################################


####################################################
def eval_yzplane(rdm0, x, boundy, boundz, yzorig=(0.0, 0.0), mol=None, tdir=None, 
                 orb=None, nCore=None, nCAS=None, nelCAS=None, tnorm0=False, tnorm1=True, 
                 prefix='plane_hole', print_cart=False, simtime_thr=1E-11, verbose=2,
                 logbook=None):

    assert len(yzorig) == 2
    orig = (0.0, yzorig[0], yzorig[1])
    boundz0 = (-boundz[2], boundz[1], -boundz[0])
    eval_plane(rdm0, (1.0, 0.0, 0.0), x, orig, boundz0, boundy, 0.0, mol, tdir, orb, 
               nCore, nCAS, nelCAS, tnorm0, tnorm1, prefix, print_cart, simtime_thr, 
               verbose, logbook)
####################################################


####################################################
def eval_plane(rdm0, uvec, disp, orig, bound1, bound2, roll=0.0, mol=None, tdir=None, 
               orb=None, nCore=None, nCAS=None, nelCAS=None, tnorm0=False, tnorm1=True, 
               prefix='plane_hole', print_cart=False, simtime_thr=1E-11, verbose=2,
               logbook=None, rem=False):

    '''
    Evaluates hole density in a plane at each time point and then prints it to
    a file for later plotting.

    nelCAS:
       Number of active electrons in the time-dependent wavefunction (the wavefunction  
       that the cation RDM1 loaded from tdir is calculated from).
    orb:
       Active orbitals.
    uvec:
      The unit vector pointing from the origin and that is perpendicular to the 
      evaluation plane.
    disp:
      Distance in angstrom of the evaluation plane from origin.
    orig:
      Define a new origin (in angstrom) where uvec and disp are defined instead of the one 
      inferred from atomic coordinates.
    bound1: 
      The x coordinate of the evaluation points. Given as a tuple of (xl, nx, xr)
      where xl and xr are the limits of the axis and nx is the number of points
      on this axis.
    bound2:
      The y coordinate of the evaluation points. It uses the same format as bound1.
    roll:
      The amount of rotation in degrees of the evaluation plane around uvec.

    Detailed description:
      The evaluation plane is obtained by rotating a plane parallel to the xy-plane but 
      shifted at z=disp by <roll> degrees around the z axis, then rotating by alpha
      degrees around y axis, and finally rotating by beta degrees around z axis. Alpha
      is the angle between the projection of uvec on xy-plane and the x axis. Beta is 
      the angle between uvec and z-axis. The rotated plane is then uniformly shifted 
      along the vector orig.
    '''

    assert len(uvec) == 3
    assert np.linalg.norm(uvec) > 1E-12
    
    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]
    if tdir is None:
        tdir = logbook['sample_dir']        
    
    #==== Determine rotation matrix ====#
    uvec0 = uvec / np.linalg.norm(uvec)
    ax1l, nax1, ax1r = bound1
    ax2l, nax2, ax2r = bound2
    ax1 = ang2bohr * np.linspace(ax1l, ax1r, nax1)
    ax2 = ang2bohr * np.linspace(ax2l, ax2r, nax2)
    ax3 = ang2bohr * disp
    if uvec0[1] >= 0.0:
        alpha = np.arctan2(uvec0[1], uvec0[0]) * rad2deg
    else:
        alpha = 360.0 + np.arctan2(uvec0[1], uvec0[0])*rad2deg
    beta = np.arccos(uvec0[2]) * rad2deg
    rot = util_general.rotmat(alpha, beta, roll)

    #==== Get evaluation points on the plane ====#
    print('alpha = %.4f' % alpha)
    print('beta = %.4f' % beta)
    coords = np.zeros((nax1*nax2, 3))
    co = 0
    for i1 in range(nax1):
        for i2 in range(nax2):
            c = np.array( [ax1[i1], ax2[i2], ax3] )
            coords[co,:] = rot @ c + orig
            co += 1

    #==== Evaluate active orbitals at the above printing range ====#
    ao_eval = mol.eval_gto("GTOval_sph", coords)
    orb_eval = ao_eval @ orb

    #==== Construct time vector ====#
    tt, _, ntu, rdm1_dir = util_general.extract_tevo(tdir)
    idsort = np.argsort(tt)
    ndigit = len(str(ntu))

    #==== RDM1 of neutral ====#
    if tnorm0:
        tr = np.sum( np.trace(rdm0, axis1=1, axis2=2) ).real
        rdm0 = np.sum(rdm0, axis=0) * (nelCAS+1) / tr
    else:
        rdm0 = np.sum(rdm0, axis=0)

    #==== Calculate and print hole density ====#
    k = 0
    kk = 0
    for i in idsort:
        if kk > 0:
            assert not (tt[i] < t_last), 'Time points are not properly sorted, this is ' \
                'a bug in the program. Report to the developer. ' + \
                f'Current time point = {tt[i]:13.8f}.'

        #==== Remove existing *.tdh files ====#
        if rem:
            oldfiles = glob.glob(rdm1_dir[i] + '/*' + EXT1)
            for f in oldfiles: os.remove(f)
            
        #==== Load cation RDM1 ====#
        rdm1 = np.load(rdm1_dir[i] + '/1pdm.npy')
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
                print('    Cation RDM1 loaded from ' + rdm1_dir[i])
                print('    Trace of the loaded cation 1RDM = ' +
                      '%12.8f (Re), %12.8f (Im)' % (tr.real, tr.imag))
            echeck = np.linalg.eigvalsh(rdm1)
                        
            #==== Calculate and print hole density function ====#
            ut_name = rdm1_dir[i] + '/' + prefix + '-' + str(k).zfill(ndigit) + EXT1
            if verbose > 1:
                print('    Printing on-plane hole density into ' + ut_name)
            with open(ut_name, 'w') as outf:
                outf.write('# %16s  %16s' % ('axis1 (angstrom)', 'axis2 (angstrom)'))
                if print_cart:
                    outf.write('  %16s  %16s  %16s' %
                           ('x (angstrom)', 'y (angstrom)', 'z (angstrom)'))
                outf.write('     %22s\n' % 'hole density')
                co = 0
                for i1 in range(0,nax1):
                    for i2 in range(0,nax2):
                        hdens_core = 0  #2 * np.dot(orb_eval[co,0:nCore], orb_eval[co,0:nCore])
                        hdens_cas = orb_eval[co,:] @ (rdm0-rdm1) @ orb_eval[co,:]
                        hdens = hdens_core + hdens_cas.real
                        ax1_, ax2_ = ax1[i1]/ang2bohr, ax2[i2]/ang2bohr
                        xa, ya, za = coords[co,:]/ang2bohr
                        outf.write('  %16.6f  %16.6f' % (ax1_, ax2_))
                        if print_cart:
                            outf.write('  %16.6f  %16.6f  %16.6f' % (xa, ya, za))
                        outf.write('     %22.14e\n' % hdens.real)
                        co += 1
                    outf.write('\n')

            #==== Increment unique time point index ====#
            k += 1
            
        #==== Duplicate time point ====#
        elif kk > 0 and tt[i]-t_last < simtime_thr:
            util_print.print_warning('The data loaded from \n    ' + rdm1_dir[i] +
                                     '\nhas a time point identical to the previous ' + \
                                     f'time point. Duplicate time point = {tt[i]:13.8f}')
            echeck_tsim = np.linalg.eigvalsh(rdm1)
            if max(np.abs(echeck_tsim - echeck)) > 1E-6:
                util_print.print_warning\
                    (f'The 1RDM loaded at the identical time point {tt[i]:13.8f} yields ' +
                     'eigenvalues different by more than 1E-6 as the other identical \n' +
                     'time point. Ideally you don\'t want to have such inconsistency ' +
                     'in your data. Proceed at your own risk.')
            else:
                print('   Data at this identical time point is consistent with the previous ' + \
                      'time point. This is good.\n')
                
        t_last = tt[i]

        #==== Increment general (non-unique) time point index ====#
        kk += 1
####################################################


####################################################
def eval_plane_orb(rdm0, uvec, disp, orig, bound1, bound2, orbx, roll=0.0, mol=None, 
                   tdir=None, orb=None, nCore=None, nCAS=None, nelCAS=None, 
                   tnorm0=False, tnorm1=True, prefix='plane_hole_orb', print_cart=False, 
                   simtime_thr=1E-11, verbose=2, logbook=None, rem=False):

    '''
    Evaluates hole density in a plane at each time point using density matrix 
    that has been projected to the space spanned by orbx and then prints it to
    a file for later plotting.
    '''
    
    if mol is None:
        mol = util_atoms.mole(logbook)
    ovl = mol.intor('int1e_ovlp')
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nCore is None:
        nCore = logbook['nCore']
    if orb is None:
        orb = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]

    f = orbx @ orbx.T @ ovl @ orb

    eval_plane(rdm0, uvec, disp, orig, bound1, bound2, roll, mol, tdir, f, nCore,
               nCAS, nelCAS, tnorm0, tnorm1, prefix, print_cart, simtime_thr, 
               verbose, logbook, rem)
####################################################


####################################################
def eval_volume(rdm0, nc, mol=None, tdir=None, orb=None, nCore=None, nCAS=None, 
                nelCAS=None, tnorm0=False, tnorm1=True, prefix='volume_hole', 
                simtime_thr=1E-11, verbose=2, logbook=None, rem=False):

    '''
    Evaluates the hole density in 3D space at each time point and then prints it
    to a cube file for later plotting.

    Args:
      rdm0: numpy.ndarray of shape (2,nCAS,nCAS)
        The 1RDM of the ground state of the neutral molecule in site orbitals
        basis. The first dimension corresponds to the spin channel.
      nc: 3-element tuple
        Specifies the number of points to evaluate the hole density in space.

    Kwargs:
      mol: Mole object
        Mole object describing the molecule.
      tdir: tuple of strings
        The strings are the paths to the different parts of the
        simulation. There can be more than one part if there are restarts.
      orb: numpy.ndarray of shape (mol.nao, nCore+nCAS)
        The AO coefficients of active orbitals. The rows correspond to the AO
        index while the columns correspond to the orbital index.
      nCore: int
        The number of core orbitals contained making up the first nCore columns
        of orb.
      nCAS: int
        The number of active orbitals.
      nelCAS: int
        The number of active electrons.
      tnorm0: bool
        If True, rdm0 will be normalized.
      tnorm1: bool
        If True, the time-dependent 1RDM of the cation at each time point will
        be normalized.
      prefix: string
        The prefix for the generated cube files.
      simtime_thr: float
        The threshold to determine if two subsequent time points are identical.
      verbose: int
        Verbosity level of the ouput.
      logbook: dict
        Logbook file from a previous TDDMRG simulation. If mol, nCore, nCAS,
        nelCAS, orb, and tdir are None or not given, logbook must be given from
        which their values will be extracted.
    '''
    assert len(nc) == 3
    
    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]
    if tdir is None:
        tdir = logbook['sample_dir']        
    osym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orb)

    #==== Construct time vector ====#
    tt, _, ntu, rdm1_dir = util_general.extract_tevo(tdir)
    idsort = np.argsort(tt)
    ndigit = len(str(ntu))
    
    #==== RDM1 of neutral in pseudo-AO ====#
    if tnorm0:
        tr = np.sum( np.trace(rdm0, axis1=1, axis2=2) ).real
        rdm0 = np.sum(rdm0, axis=0) * (nelCAS+1) / tr
    else:
        rdm0 = np.sum(rdm0, axis=0)
    rdm0_ao = orb @ rdm0 @ orb.T      # rdm0_ao is the RDM in pseudo-AO rep.

    #==== Calculate and print hole density ====#
    k = 0
    kk = 0
    for i in idsort:
        if kk > 0:
            assert not (tt[i] < t_last), 'Time points are not properly sorted, this is ' \
                'a bug in the program. Report to the developer. ' + \
                f'Current time point = {tt[i]:13.8f}.'

        #==== Remove existing *.tdh files ====#
        if rem:
            oldfiles = glob.glob(rdm1_dir[i] + '/*' + EXT2)
            for f in oldfiles: os.remove(f)
            
        #==== Load cation RDM1 ====#
        rdm1 = np.load(rdm1_dir[i] + '/1pdm.npy')
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
                print('    Cation RDM1 loaded from ' + rdm1_dir[i])
                print('    Trace of the loaded cation 1RDM = ' +
                      '%12.8f (Re), %12.8f (Im)' % (tr.real, tr.imag))
            echeck = np.linalg.eigvalsh(rdm1)
                        
            #==== RDM1 of cation in pseudo-AO basis ====#
            rdm1_ao = orb @ rdm1 @ orb.T

            #==== Print to cube files ====#
            cubename = rdm1_dir[i] + '/' + prefix + '-' + str(k).zfill(ndigit) + EXT2
            if verbose > 1:
                print('    Printing hole density into cube file ' + cubename)
            tools.cubegen.density(mol, cubename, rdm0_ao-rdm1_ao, nx=nc[0], ny=nc[1],
                                  nz=nc[2])

            #==== Increment unique time point index ====#
            k += 1
            
        #==== Duplicate time point ====#
        elif kk > 0 and tt[i]-t_last < simtime_thr:
            util_print.print_warning\
                ('The data loaded from \n    ' + rdm1_dir[i] + '\nhas a time point ' + \
                 'identical to the previous time point. ' + \
                 f'Duplicate time point = {tt[i]:13.8f}')
            echeck_tsim = np.linalg.eigvalsh(rdm1)
            if max(np.abs(echeck_tsim - echeck)) > 1E-6:
                util_print.print_warning\
                    (f'The 1RDM loaded at the identical time point {tt[i]:13.8f} yields ' +
                     'eigenvalues different by more than 1E-6 as the other identical \n' +
                     'time point. Ideally you don\'t want to have such inconsistency ' +
                     'in your data. Proceed at your own risk.')
            else:
                print('   Data at this identical time point is consistent with the previous ' + \
                      'time point. This is good.\n')
                
        t_last = tt[i]

        #==== Increment general (non-unique) time point index ====#
        kk += 1
####################################################


####################################################
def eval_volume_orb(rdm0, nc, orbx, mol=None, tdir=None, orb=None, nCore=None, 
                    nCAS=None, nelCAS=None, tnorm0=False, tnorm1=True, 
                    prefix='volume_hole_orb', simtime_thr=1E-11, verbose=2, 
                    logbook=None, rem=False):

    '''
    Evaluates hole density in 3D space at each time point using density matrix 
    that has been projected to the space spanned by orbx and then prints it to
    a cube file for later plotting.
    '''
    
    if mol is None:
        mol = util_atoms.mole(logbook)
    ovl = mol.intor('int1e_ovlp')
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nCore is None:
        nCore = logbook['nCore']
    if orb is None:
        orb = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]

    f = orbx @ orbx.T @ ovl @ orb

    eval_volume(rdm0, nc, mol, tdir, f, nCore, nCAS, nelCAS, tnorm0, tnorm1,
                prefix, simtime_thr, verbose, logbook, rem)
    
####################################################



            
###==== Setting up the system ====#
#mol = gto.M(atom=atoms, basis=basis, ecp=ecp,
#            symmetry=group)
#na, nb = mol.nelec
#n_mo = mol.nao
#nocc = nCore + nCAS
#
#print('No. of bases = ', mol.nao)
#print('No. of electrons (alpha, beta) = ', na, nb)
#print('No. of core orbitals = ', nCore)
#print('No. of CAS orbitals = ', nCAS)
#mo_c = np.load(orb_path)
#
#
##==== Evaluate MO's at the above printing range ====#
#ao_value = mol.eval_gto("GTOval_sph", coords)
#mo_value = ao_value @ mo_c
#print('AO_value shape = ', ao_value.shape)
#print('MO_coeff shape = ', mo_c.shape)
#print('MO_value shape = ', mo_value.shape)
#
#
## ==== Evaluate the hole density (input the PDM's in MO basis here) ====#
#pdm0 = np.zeros((2, n_mo, n_mo), dtype=complex)
#pdm1 = np.zeros((2, n_mo, n_mo), dtype=complex)
#for i in range(0,2):
#    for j in range(0, nCore): pdm0[i,j,j] = complex(1.0, 0.0)
#    for j in range(0, nCore): pdm1[i,j,j] = complex(1.0, 0.0)
#    
#
#pdm0_ = np.load(pdmref_path)
#for i in range(0,2): pdm0[i, nCore:nocc, nCore:nocc] = pdm0_[i,:,:]
#pdm0_ao = mo2ao_l @ np.sum(pdm0, axis=0) @ mo2ao_r
#ndigit = len(str(n_sample))
#for ifl in range(0, n_sample):
#    pdm1_dir = './' + prefix + '.sample/tevo-' + str(ifl)
#    pdm1_ = np.load(pdm1_dir + '/1pdm.npy')
#    for i in range(0,2): pdm1[i, nCore:nocc, nCore:nocc] = pdm1_[i,:,:]
#    print('Time point ', ifl)
#    print('  Path = ' + pdm1_dir)
#    tr = np.trace(np.sum(pdm1, axis=0))
#    print('  Trace of 1RDM before normalization = %12.8f (Re), %12.8f (Im)' % (tr.real, tr.imag))
#
#    nel = np.trace(pdm1[0,:,:] + pdm1[1,:,:])
#    nel_ref = na + nb - 1
#    pdm1 = pdm1/nel * nel_ref
#    tr = np.trace(np.sum(pdm1, axis=0))
#    print('  Trace of 1RDM after normalization =  %12.8f (Re), %12.8f (Im)' % (tr.real, tr.imag))
#    
#    with open(pdm1_dir + '/dens_fun', 'w') as densplot:
#        co = 0
#        densplot.write('%12s  %12s  %12s  %22s\n' %
#                       ('x (angstrom)', 'y (angstrom)', 'z (angstrom)', 'hole density'))
#        for i in range(0,nx):
#            for j in range(0,ny):
#                for k in range(0,nz):
#                    densf = np.einsum('ij, i, j', np.sum(pdm0-pdm1, axis=0), mo_value[co,:],
#                                      mo_value[co,:])
#                    x_, y_, z_ = x[i]/ang2bohr, y[j]/ang2bohr, z[k]/ang2bohr
#                    densplot.write('%12.4f  %12.4f  %12.4f  %22.14e\n' %
#                                  (x_, y_, z_, densf.real))
#                    co = co + 1
#    
#                if nx == 1: densplot.write(' \n')
#            if ny == 1 or nz == 1: densplot.write(' \n')
#    print(' ')
#
#    #==== Print into cube files ====#
#    if make_cube:
#        pdm1_ao = mo2ao_l @ np.sum(pdm1, axis=0) @ mo2ao_r
#        hole_ao = pdm0_ao - pdm1_ao
#        filename = pdm1_dir + '/hole_' + str(ifl).zfill(ndigit) + '.cube'
#        cubegen.density(mol, filename, hole_ao, nx=nxc, ny=nyc, nz=nzc)
#
