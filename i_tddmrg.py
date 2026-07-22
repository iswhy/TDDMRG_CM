
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""
A class for the analysis of charge migration starting from
an initial state produced by the application of the annihilation 
operator on a given site/orbital.

Original version:
     Imam Wahyutama, May 2023
Derived from:
     gfdmrg.py
     ft_tddmrg.py
"""

#from ipsh import ipsh

import os, time, glob
from humanfriendly import format_timespan
import numpy as np
import scipy.linalg
from scipy.linalg import eigvalsh, eigh

# Set spin-adapted or non-spin-adapted here
spin_symmetry = 'su2'
#spin_symmetry = 'sz'

import block2 as b2
from TDDMRG_CM.utils.util_complex_type import get_complex_type
comp = get_complex_type()
if comp == 'full':
    bx = b2.cpx
    bc = bx
elif comp == 'hybrid':
    bx = b2
    bc = None    #OLD block2.cpx if has_cpx else None

if spin_symmetry == 'su2':
    bs = bx.su2
    brs = b2.su2
    SX = b2.SU2
elif spin_symmetry == 'sz':
    bs = bx.sz
    brs = b2.sz
    SX = b2.SZ

try:
    if spin_symmetry == 'su2':
        from block2.su2 import MPICommunicator
    elif spin_symmetry == 'sz':
        from block2.sz import MPICommunicator
    hasMPI = True
except ImportError:
    hasMPI = False



#OLD_CPX if SpinLabel == SU2:
#OLD_CPX     from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
#OLD_CPX     from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, IdentityMPO
#OLD_CPX     from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
#OLD_CPX     from block2.su2 import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
#OLD_CPX     try:
#OLD_CPX         from block2.su2 import MPICommunicator
#OLD_CPX         hasMPI = True
#OLD_CPX     except ImportError:
#OLD_CPX         hasMPI = False
#OLD_CPX else:
#OLD_CPX     from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
#OLD_CPX     from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, IdentityMPO
#OLD_CPX     from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
#OLD_CPX     from block2.sz import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
#OLD_CPX     try:
#OLD_CPX         from block2.sz import MPICommunicator
#OLD_CPX         hasMPI = True
#OLD_CPX     except ImportError:
#OLD_CPX         hasMPI = False

from pyblock2 import tools as b2tools; b2tools.init(SX)
#OLD import tools; tools.init(SX)
#from tools import saveMPStoDir, mkDir
#OLD from tools import mkDir
from TDDMRG_CM.utils.util_print import getVerbosePrinter
from TDDMRG_CM.utils.util_print import print_section, print_warning, print_describe_content, print_matrix
from TDDMRG_CM.utils.util_print import print_orb_occupations, print_pcharge, print_mpole, print_bond_order
from TDDMRG_CM.utils.util_print import print_autocorrelation, print_td_pcharge, print_td_bo, \
    print_td_mpole
from TDDMRG_CM.utils.util_qm import make_full_dm
from TDDMRG_CM.utils.util_mps import print_MPO_bond_dims, MPS_fitting, calc_energy_MPS
from TDDMRG_CM.utils.util_mps import saveMPStoDir, loadMPSfromDir_OLD, loadMPSfromDir
from TDDMRG_CM.utils.util_mps import trans_to_singlet_embed
from TDDMRG_CM.utils import util_logbook
from TDDMRG_CM.observables import pcharge, mpole, bond_order
from TDDMRG_CM.phys_const import au2fs

if hasMPI:
    MPI = MPICommunicator()
    r0 = (MPI.rank == 0)
else:
    class _MPI:
        rank = 0
    MPI = _MPI()
    r0 = True
    
_print = getVerbosePrinter(r0, flush=True)
print_i2 = getVerbosePrinter(r0, indent=2*' ', flush=True)
print_i4 = getVerbosePrinter(r0, indent=4*' ', flush=True)
    
    


#################################################
class MYTDDMRG:
    """
    DDMRG++ for Green's Function for molecules.
    """


    #################################################
    def __init__(self, mol, nel_site, scratch='./nodex', memory=1*1E9, isize=6E8, 
                 omp_threads=8, verbose=2, print_statistics=True, mpi=None,
                 delayed_contraction=True):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """


        if spin_symmetry == 'sz':
            _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            _print('WARNING: SZ Spin label is chosen! The MYTDDMRG class was designed ' +
                   'with the SU2 spin label in mind. The use of SZ spin label in this ' +
                   'class has not been checked.')
            _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        _print('Memory = %10.2f Megabytes' % (memory/1.0e6))
        _print('Integer size = %10.2f Megabytes' % (isize/1.0e6))

        
        b2.Random.rand_seed(0)
        isize = int(isize)
        assert isize < memory
        #OLD isize = min(int(memory * 0.1), 200000000)
        b2.init_memory(isize=isize, dsize=int(memory - isize), save_dir=scratch)
        b2.Global.threading = b2.Threading(
            b2.ThreadingTypes.OperatorBatchedGEMM | b2.ThreadingTypes.Global, omp_threads,
            omp_threads, 1)
        b2.Global.threading.seq_type = b2.SeqTypes.Tasked
        b2.Global.frame.load_buffering = False
        b2.Global.frame.save_buffering = False
        b2.Global.frame.use_main_stack = False
        b2.Global.frame.minimal_disk_usage = True

        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.print_statistics = print_statistics
        self.mpi = mpi
        ## self.mpi = MPI
        
        self.delayed_contraction = delayed_contraction
        self.idx = None # reorder
        self.ridx = None # inv reorder
        if self.mpi is not None:
            print('herey2 I am MPI', self.mpi.rank)
            self.mpi.barrier()

        
        #==== Create scratch directory ====#
        if self.mpi is not None:
            if self.mpi.rank == 0:
                b2tools.mkDir(scratch)
            self.mpi.barrier()
        else:
            b2tools.mkDir(scratch)

        if self.verbose >= 2:
            _print(b2.Global.frame)
            _print(b2.Global.threading)

        if mpi is not None:
            #OLD_CPX if SpinLabel == SU2:
            #OLD_CPX     from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelRuleSiteQC
            #OLD_CPX     from block2.su2 import ParallelRuleSiteQC, ParallelRuleIdentity
            #OLD_CPX else:
            #OLD_CPX     from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
            #OLD_CPX     from block2.sz import ParallelRuleSiteQC, ParallelRuleIdentity
            self.prule = bs.ParallelRuleQC(mpi)
            self.pdmrule = bs.ParallelRuleNPDMQC(mpi)
            self.siterule = bs.ParallelRuleSiteQC(mpi)
            self.identrule = bs.ParallelRuleIdentity(mpi)
        else:
            self.prule = None
            self.pdmrule = None
            self.siterule = None
            self.identrule = None

        #==== Some persistent quantities ====#
        self.mol = mol
        #OLDassert isinstance(nel_site, tuple), 'init: The argument nel_site must be a tuple.'
        self.nel = sum(mol.nelec)
        self.nel_site = nel_site
        self.nel_core = self.nel - nel_site
        assert self.nel_core%2 == 0, \
            f'The number of core electrons (currently {self.nel_core}) must be an even ' + \
            'number.'
        self.ovl_ao = mol.intor('int1e_ovlp')
        mol.set_common_origin([0,0,0])
        self.dpole_ao = mol.intor('int1e_r').reshape(3,mol.nao,mol.nao)
        self.qpole_ao = mol.intor('int1e_rr').reshape(3,3,mol.nao,mol.nao)
            
    #################################################


    #################################################
    def assign_orbs(self, n_core, n_sites, orbs):

        if spin_symmetry == 'su2':
            n_mo = orbs.shape[1]
            #OLDassert orbs.shape[0] == orbs.shape[1]
        elif spin_symmetry == 'sz':
            n_mo = orbs.shape[2]
            #OLDassert orbs.shape[1] == orbs.shape[2]
        assert n_mo <= self.mol.nao
        n_occ = n_core + n_sites
        n_virt = n_mo - n_occ

        if spin_symmetry == 'su2':
            assert len(orbs.shape) == 2, \
                'If SU2 symmetry is invoked, orbs must be a 2D array.'
            orbs_c = np.zeros((2, orbs.shape[0], n_core))
            orbs_s = np.zeros((2, orbs.shape[0], n_sites))
            orbs_v = np.zeros((2, orbs.shape[0], n_virt))
            for i in range(0,2):
                orbs_c[i,:,:] = orbs[:, 0:n_core]
                orbs_s[i,:,:] = orbs[:, n_core:n_occ]
                orbs_v[i,:,:] = orbs[:, n_occ:n_mo]
        elif spin_symmetry == 'sz':
            assert len(orbs.shape) == 3 and orbs.shape[0] == 2, \
                'If SZ symmetry is invoked, orbs must be a 3D array with the size ' + \
                'of first dimension being two.'
            orbs_c = orbs[:, :, 0:n_core]
            orbs_s = orbs[:, :, n_core:n_occ]
            orbs_v = orbs[:, :, n_occ:n_mo]

        return orbs_c, orbs_s, orbs_v
    #################################################

            
    #################################################
    def init_hamiltonian_fcidump(self, pg, filename, orbs, idx=None):
        """Read integrals from FCIDUMP file."""
        assert self.fcidump is None
        self.fcidump = bx.FCIDUMP()
        self.fcidump.read(filename)
        self.groupname = pg
        assert self.fcidump.n_elec == self.nel_site, \
            f'init_hamiltonian_fcidump: self.fcidump.n_elec ({self.fcidump.n_elec}) must ' + \
            'be identical to self.nel_site (%d).' % self.nel_site

        #==== Reordering indices ====#
        if idx is not None:
            self.fcidump.reorder(b2.VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)

        #==== Orbitals and MPS symemtries ====#
        swap_pg = getattr(b2.PointGroup, "swap_" + pg)
        self.orb_sym = b2.VectorUInt8(map(swap_pg, self.fcidump.orb_sym))      # 1)
        _print("# fcidump symmetrize error:", self.fcidump.symmetrize(orb_sym))
        self.wfn_sym = swap_pg(self.fcidump.isym)
        # NOTE:
        # 1) Because of the self.fcidump.reorder invocation above, self.orb_sym contains
        #    orbital symmetries AFTER REORDERING.

        #==== Construct the Hamiltonian MPO ====#
        vacuum = SX(0)
        self.target = SX(self.fcidump.n_elec, self.fcidump.twos, swap_pg(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites
        self.hamil = bs.HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        #==== Assign orbitals ====#
        self.core_orbs, self.site_orbs, self.virt_orbs = \
            self.assign_orbs(int(self.nel_core/2), self.n_sites, orbs)
        self.n_core, self.n_virt = self.core_orbs.shape[2], self.virt_orbs.shape[2]
        self.n_orbs = self.n_core + self.n_sites + self.n_virt

        #==== Reordering orbitals ====#
        if idx is not None:
            self.site_orbs = self.site_orbs[:,:,idx]
    #################################################


    #################################################
    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym, e_core, 
                         h1e, g2e, orbs, tol=1E-13, idx=None, save_fcidump=None):
        """
        Initialize integrals using h1e, g2e, etc.
        n_elec : The number of electrons within the sites. This means, if there are core
                 orbitals, then n_elec are the number of electrons in the active space only.
        isym: wfn symmetry in molpro convention? See the getFCIDUMP function in CAS_example.py.
        g2e: Does it need to be in 8-fold symmetry? See the getFCIDUMP function in CAS_example.py.
        orb_sym: orbitals symmetry in molpro convention.
        """

        #==== Initialize self.fcidump ====#
        assert self.fcidump is None
        self.fcidump = bx.FCIDUMP()
        self.groupname = pg
        assert n_elec == self.nel_site, \
            f'init_hamiltonian: The argument n_elec ({n_elec}) must be identical to ' + \
            'self.nel_site (%d).' % self.nel_site

        #==== Rearrange the 1e and 2e integrals, and initialize FCIDUMP ====#
        if not isinstance(h1e, tuple):
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol, '\n' + \
                        f'   h1e[i,j] = {h1e[i,j]:17.13f} \n' + \
                        f'   h1e[j,i] = {h1e[j,i]:17.13f} \n' + \
                        f'   Delta = {h1e[i, j] - h1e[j, i]:17.13f} \n' + \
                        f'   tol. = {tol:17.13f}'
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.ravel()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
            if self.verbose >= 2:
                _print('Number of 1e integrals (incl. hermiticity) = ', mh1e.size)
                _print('Number of 2e integrals (incl. hermiticity) = ', mg2e.size)
            
            self.fcidump.initialize_su2(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        else:
            assert spin_symmetry == 'sz'
            #OLDassert twos == 2*(self.nel_site[0]-self.nel_site[1]), \
            #OLD    'init_hamiltonian: When SZ symmetry is enabled, the argument twos must be ' + \
            #OLD    'equal to twice the difference between alpha and beta electrons. ' + \
            #OLD    f'Currently, their values are twos = {twos} and 2*(n_alpha - n_beta) = ' + \
            #OLD    f'{2*(self.nel_site[0]-self.nel_site[1])}.'
            assert isinstance(h1e, tuple) and len(h1e) == 2
            assert isinstance(g2e, tuple) and len(g2e) == 3
            mh1e_a = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e_b = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e = (mh1e_a, mh1e_b)
            for xmh1e, xh1e in zip(mh1e, h1e):
                k = 0
                for i in range(0, n_sites):
                    for j in range(0, i + 1):
                        assert abs(xh1e[i, j] - xh1e[j, i]) < tol
                        xmh1e[k] = xh1e[i, j]
                        k += 1
                xmh1e[np.abs(xmh1e) < tol] = 0.0
            #OLD mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            mg2e = tuple(xg2e.ravel() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0      # xmg2e works like a pointer to the elements of mg2e tuple.
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)

        #==== Take care of the symmetry conventions. Note that ====#
        #====   self.fcidump.orb_sym is in Molpro convention,  ====#
        #====     while self.orb_sym is in block2 convention   ====#
        self.fcidump.orb_sym = b2.VectorUInt8(orb_sym)       # Hence, self.fcidump.orb_sym is in Molpro convention.

        #==== Reordering indices ====#
        if idx is not None:
            self.fcidump.reorder(b2.VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)

        #==== Orbitals and MPS symemtries ====#
        swap_pg = getattr(b2.PointGroup, "swap_" + pg)
        self.orb_sym = b2.VectorUInt8(map(swap_pg, self.fcidump.orb_sym))      # 1)
        self.wfn_sym = swap_pg(isym)
        # NOTE:
        # 1) Because of the self.fcidump.reorder invocation above, self.orb_sym contains
        #    orbital symmetries AFTER REORDERING.


        #==== Construct the Hamiltonian MPO ====#
        vacuum = SX(0)
        self.target = SX(n_elec, twos, swap_pg(isym))
        self.n_sites = n_sites
        self.hamil = bs.HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        #==== Assign orbitals ====#
        self.core_orbs, self.site_orbs, self.virt_orbs = \
            self.assign_orbs(int(self.nel_core/2), self.n_sites, orbs)
        self.n_core, self.n_virt = self.core_orbs.shape[2], self.virt_orbs.shape[2]
        self.n_orbs = self.n_core + self.n_sites + self.n_virt

        #==== Reorder orbitals ====#
        if idx is not None:
            self.site_orbs = self.site_orbs[:,:,idx]
        
        #==== Save self.fcidump ====#
        if save_fcidump is not None:
            if self.mpi is None or self.mpi.rank == 0:
                #self.fcidump.orb_sym = b2.VectorUInt8(self.orb_sym)
                self.fcidump.write(self.scratch + '/' + save_fcidump)
            if self.mpi is not None:
                self.mpi.barrier()


                

        #==== New interface ====#
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
        if spin_symmetry == 'su2': symm_type = [SymmetryTypes.SU2]
        elif spin_symmetry == 'sz': symm_type = [SymmetryTypes.SZ]
        if comp == 'full': symm_type += [SymmetryTypes.CPX]
        self.b2driver = DMRGDriver(scratch=self.scratch, symm_type=symm_type, stack_mem=4 << 30, 
                                n_threads=56, mpi=self.mpi)
        swap_pg = getattr(b2.PointGroup, "swap_" + pg)
        self.b2driver.initialize_system(n_sites=self.n_sites, n_elec=n_elec, spin=twos, 
                                     singlet_embedding=False, pg_irrep=self.wfn_sym,
                                     orb_sym=b2.VectorUInt8(map(swap_pg, orb_sym)))
        self.te_mpo = self.b2driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=e_core, reorder=idx,
                                            iprint=1)
        #self.te_mpo = self.b2driver.get_conventional_qc_mpo(self.fcidump,
        #                                                 MPOAlgorithmTypes.Conventional)
        #_print('TE_MPO algo type = ', MPOAlgorithmTypes.Conventional)
        
        if self.mpi is not None:
            self.te_mpo = bs.ParallelMPO(self.te_mpo, self.prule)


            

            
        if self.mpi is not None:
            self.mpi.barrier()
    #################################################


    #################################################
    def unordered_site_orbs(self):
        
        if self.ridx is None:
            return self.site_orbs
        else:
            return self.site_orbs[:,:,self.ridx]
    #################################################


    #################################################
    @staticmethod
    def fmt_size(i, suffix='B'):
        if i < 1000:
            return "%d %s" % (i, suffix)
        else:
            a = 1024
            for pf in "KMGTPEZY":
                p = 2
                for k in [10, 100, 1000]:
                    if i < k * a:
                        return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                    p -= 1
                a *= 1024
        return "??? " + suffix
    #################################################


    #################################################
    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, cpx_mps, mps=None, inmps_name=None, dmargin=0):
        if mps is None and inmps_name is None:
            raise ValueError("The 'mps' and 'inmps_name' parameters of "
                             + "get_one_pdm cannot be both None.")
        
        if self.verbose >= 2:
            _print('>>> START one-pdm <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        if mps is None:   # mps takes priority over inmps_name, the latter will only be used if the former is None.
            mps_info = brs.MPSInfo(0)
            mps_info.load_data(self.scratch + "/" + inmps_name)
            mps = bs.MPS(mps_info)
            mps.load_data()
            mps.info.load_mutable()
            
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        #OLD_CPX if self.mpi is not None:
        #OLD_CPX     if SpinLabel == SU2:
        #OLD_CPX         from block2.su2 import ParallelMPO
        #OLD_CPX     else:
        #OLD_CPX         from block2.sz import ParallelMPO

        # 1PDM MPO
        pmpo = bs.PDM1MPOQC(self.hamil)
        pmpo = bs.SimplifiedMPO(pmpo, bs.RuleQC())
        if self.mpi is not None:
            pmpo = bs.ParallelMPO(pmpo, self.pdmrule)

        # 1PDM
        pme = bs.MovingEnvironment(pmpo, mps, mps, "1PDM")
        pme.init_environments(False)
        if cpx_mps and comp == 'hybrid':
            # WARNING: There is no ComplexExpect in block2.cpx.su2
            expect = brs.ComplexExpect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
        else:
            expect = bs.Expect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, mps.center == 0)
        if spin_symmetry == 'su2':
            dmr = expect.get_1pdm_spatial(self.n_sites)
            dm = np.array(dmr).copy()
        elif spin_symmetry == 'sz':
            dmr = expect.get_1pdm(self.n_sites)
            dm = np.array(dmr).copy()
            dm = dm.reshape((self.n_sites, 2,
                             self.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))

        if self.ridx is not None:
            dm[:, :] = dm[self.ridx, :][:, self.ridx]

        mps.save_data()
        if mps is None:
            mps_info.deallocate()
        dmr.deallocate()
        pmpo.deallocate()

        if self.verbose >= 2:
            _print('>>> COMPLETE one-pdm | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        if spin_symmetry == 'su2':
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        elif spin_symmetry == 'sz':
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
    #################################################


    #################################################
    def get_trans_one_pdm(self, cpx, bMPS, kMPS, dmargin=0):

        bmps = bMPS.deep_copy('bmps')
        kmps = kMPS.deep_copy('kmps')
        
        if self.verbose >= 2:
            _print('>>> START trans one-pdm <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        #OLDif bmps is None:   # bmps takes priority over bmps_name, the latter will only be used if the former is None.
        #OLD    bmps_info = brs.MPSInfo(0)
        #OLD    bmps_info.load_data(self.scratch + "/" + bmps_name)
        #OLD    bmps = bs.MPS(bmps_info)
        #OLD    bmps.load_data()
        #OLD    bmps.info.load_mutable()
            
        max_bdim = max([x.n_states_total for x in bmps.info.left_dims])
        if bmps.info.bond_dim < max_bdim:
            bmps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in bmps.info.right_dims])
        if bmps.info.bond_dim < max_bdim:
            bmps.info.bond_dim = max_bdim

        max_bdim = max([x.n_states_total for x in kmps.info.left_dims])
        if kmps.info.bond_dim < max_bdim:
            kmps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in kmps.info.right_dims])
        if kmps.info.bond_dim < max_bdim:
            kmps.info.bond_dim = max_bdim

        # 1PDM MPO
        pmpo = bs.PDM1MPOQC(self.hamil)
        pmpo = bs.SimplifiedMPO(pmpo, bs.RuleQC())
        if self.mpi is not None:
            pmpo = bs.ParallelMPO(pmpo, self.pdmrule)

        # 1PDM
        assert bmps.center == 0 and kmps.center == 0
        pme = bs.MovingEnvironment(pmpo, bmps, kmps, "1PTDM")
        pme.init_environments(False)
        if cpx and comp == 'hybrid':
            # WARNING: There is no ComplexExpect in block2.cpx.su2
            expect = brs.ComplexExpect(pme, bmps.info.bond_dim+dmargin, kmps.info.bond_dim+dmargin)   #NOTE
        else:
            expect = bs.Expect(pme, bmps.info.bond_dim+dmargin, kmps.info.bond_dim+dmargin)   #NOTE
        _print('hereo2')
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, kmps.center == 0)
        _print('hereo3')
        if spin_symmetry == 'su2':
            dmr = expect.get_1pdm_spatial(self.n_sites)
            dm = np.array(dmr).copy()
        elif spin_symmetry == 'sz':
            dmr = expect.get_1pdm(self.n_sites)
            dm = np.array(dmr).copy()
            dm = dm.reshape((self.n_sites, 2, self.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))

        if self.ridx is not None:
            dm[:, :] = dm[self.ridx, :][:, self.ridx]

        dmr.deallocate()
        pmpo.deallocate()
        pme.remove_partition_files()

        if self.verbose >= 2:
            _print('>>> COMPLETE trans one-pdm | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        if spin_symmetry == 'su2':
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        elif spin_symmetry == 'sz':
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
    #################################################


    #################################################
    def dmrg(self, logbook_in, bond_dims, noises, n_steps=30, dav_tols=1E-5, conv_tol=1E-7, 
             cutoff=1E-14, occs=None, bias=1.0, outmps_dir0=None, outmps_name='GS_MPS_INFO',
             save_1pdm=False, flip_spect=False, mrci_info=None, inmps_dir=None,
             inmps_name='mps_info.bin', out_cpx=False):
        
        """Ground-State DMRG."""
        logbook = logbook_in.copy()

        if self.verbose >= 2:
            _print('>>> START GS-DMRG <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()
        if outmps_dir0 is None:
            outmps_dir = self.scratch
        else:
            outmps_dir = outmps_dir0

        _print('Quantum number information:')
        _print(' - Input MPS = ', self.target)
        _print(' - Input MPS multiplicity = ', self.target.multiplicity)
        logbook.update({'gs:qnumber:n':self.target.n})
        logbook.update({'gs:qnumber:mult':self.target.multiplicity})
        logbook.update({'gs:qnumber:pg':self.target.pg})

        
        # MPS type (general or occupation-rectricted)
        if inmps_dir is not None:
            idMPO_ = bs.SimplifiedMPO(bs.IdentityMPO(self.hamil), bs.RuleQC(), True, True)
            if self.mpi is not None:
                idMPO_ = bs.ParallelMPO(idMPO_, self.identrule)
                
            complex_mps = (comp == 'full')
            if mrci_info is not None:
                assert mrci_info['order'] in [1, 2, 3]
                assert mrci_info['nactive2'] <= self.n_sites
                mps_type = {'type':'mrci', 'nactive2':mrci_info['nactive2'], 'order':mrci_info['order'],
                            'n_sites':self.n_sites, 'vacuum':self.hamil.vacuum, 'target':self.target,
                            'basis':self.hamil.basis}
            else:
                mps_type = {'type':'normal'}

            _print('Guess MPS for ground state DMRG will start from an old MPS located in + \n  ' +
                   inmps_dir + '/' + inmps_name)
            mps, mps_info, _ = \
                    loadMPSfromDir(inmps_dir, inmps_name, complex_mps, mps_type, idMPO_,
                                   cached_contraction=True, MPI=self.mpi, 
                                   prule=self.prule if self.mpi is not None else None)
        else:
            if mrci_info is not None:
                assert mrci_info['order'] in [1, 2, 3]
                assert mrci_info['nactive2'] <= self.n_sites
                mps_info = brs.MRCIMPSInfo(self.n_sites, 0, mrci_info['nactive2'], mrci_info['order'],
                                           self.hamil.vacuum, self.target, self.hamil.basis)
            else:
                mps_info = brs.MPSInfo(self.n_sites, self.hamil.vacuum, self.target, self.hamil.basis)
            
            mps_info.tag = 'KET'
            if occs is None:
                if self.verbose >= 2:
                    _print("Using FCI INIT MPS")
                mps_info.set_bond_dimension(bond_dims[0])
            else:
                if self.verbose >= 2:
                    _print("Using occupation number INIT MPS")
                if self.idx is not None:
                    #ERR occs = self.fcidump.reorder(VectorDouble(occs), VectorUInt16(self.idx))
                    occs = occs[self.idx]
                mps_info.set_bond_dimension_using_occ(
                    bond_dims[0], b2.VectorDouble(occs), bias=bias)
                
            mps = bs.MPS(self.n_sites, 0, 2)   # The 3rd argument controls the use of one/two-site algorithm.
            mps.initialize(mps_info)
            mps.random_canonicalize()
            
            mps.save_mutable()
            mps.deallocate()
            mps_info.save_mutable()
            mps_info.deallocate_mutable()
        

        # MPO
        tx = time.perf_counter()
        mpo = bs.MPOQC(self.hamil, b2.QCTypes.Conventional)
        mpo = bs.SimplifiedMPO(mpo, bs.RuleQC(), True, True,
                               b2.OpNamesSet((b2.OpNames.R, b2.OpNames.RD)))
        self.mpo_orig = mpo

        #==== Flip spectrum if requested ====#
        if flip_spect:
            _print('Hamiltonian spectrum will be flipped')
            mpo = -1 * mpo

        if self.mpi is not None:
            #OLD_CPX if SpinLabel == SU2:
            #OLD_CPX     from block2.su2 import ParallelMPO
            #OLD_CPX else:
            #OLD_CPX     from block2.sz import ParallelMPO
            mpo = bs.ParallelMPO(mpo, self.prule)

        if self.verbose >= 3:
            _print('MPO time = ', time.perf_counter() - tx)

        if self.print_statistics:
            _print('GS MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = brs.MPSInfo(self.n_sites, self.hamil.vacuum,
                                    self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("GS EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("GS EST PEAK MEM = ", MYTDDMRG.fmt_size(
                mem2), " SCRATCH = ", MYTDDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

            
        # DMRG
        me = bs.MovingEnvironment(mpo, mps, mps, "DMRG")
        if self.delayed_contraction:
            me.delayed_contraction = b2.OpNamesSet.normal_ops()
            me.cached_contraction = True
        tx = time.perf_counter()
        me.init_environments(self.verbose >= 4)
        if self.verbose >= 3:
            _print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = bs.DMRG(me, b2.VectorUBond(bond_dims), b2.VectorDouble(noises))
        dmrg.davidson_conv_thrds = b2.VectorDouble(dav_tols)
        dmrg.davidson_soft_max_iter = 4000
        dmrg.noise_type = b2.NoiseTypes.ReducedPerturbative
        dmrg.decomp_type = b2.DecompositionTypes.SVD
        dmrg.iprint = max(self.verbose - 1, 0)
        dmrg.cutoff = cutoff
        dmrg.solve(n_steps, mps.center == 0, conv_tol)

        self.gs_energy = dmrg.energies[-1][0]
        self.bond_dim = bond_dims[-1]
        _print("Ground state energy = %16.10f" % self.gs_energy)
        logbook.update({'gs:dmrg_energy':self.gs_energy})


        #==== MO occupations ====#
        dm0 = self.get_one_pdm(comp=='full', mps)
        dm0_full = make_full_dm(self.n_core, dm0)
        occs0 = np.zeros((2, self.n_core+self.n_sites))
        for i in range(0, 2): occs0[i,:] = np.diag(dm0_full[i,:,:]).copy().real
        print_orb_occupations(occs0)
                    
        #==== Partial charge ====#
        orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()), axis=2)
        self.qmul0, self.qlow0 = \
            pcharge.calc(self.mol, dm0_full, orbs, self.ovl_ao)
        print_pcharge(self.mol, self.qmul0, self.qlow0)
        logbook.update({'gs:mulliken':self.qmul0, 'gs:lowdin':self.qlow0})

        #==== Bond order ====#
        self.bo_mul0, self.bo_low0 = bond_order.calc(self.mol, dm0_full, orbs, self.ovl_ao)
        print_section('Mulliken bond orders', 2)
        print_bond_order(self.bo_mul0)
        print_section('Lowdin bond orders', 2)
        print_bond_order(self.bo_low0)

        #==== Multipole analysis ====#
        e_dpole, n_dpole, e_qpole, n_qpole = \
            mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm0_full, orbs)
        print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)
        logbook.update({'gs:e_dipole':e_dpole, 'gs:n_dipole':n_dpole,
                        'gs:e_quadpole':e_qpole, 'gs:n_quadpole':n_qpole})

        #==== Conversion to full complex MPS ====#
        if out_cpx:
            assert comp != 'full'
            _print('Converting the converged GS MPS to a complex MPS ...')
            mps = self.b2driver.mps_change_complex(mps, "CPX")
            
        #==== Save the output MPS ====#
        #OLD mps.save_data()
        #OLD mps_info.save_data(self.scratch + "/GS_MPS_INFO")
        #OLD mps_info.deallocate()
        _print('')
        _print('Saving the ground state MPS files under ' + outmps_dir)
        if outmps_dir != self.scratch:
            b2tools.mkDir(outmps_dir)
        mps_info.save_data(outmps_dir + "/" + outmps_name)
        saveMPStoDir(mps, outmps_dir, self.mpi)
        _print('Output ground state max. bond dimension = ', mps.info.bond_dim)
        if save_1pdm:
            np.save(outmps_dir + '/GS_1pdm', dm0)


        #==== Statistics ====#
        if self.print_statistics:
            dmain, dseco, imain, iseco = b2.Global.frame.peak_used_memory
            _print("GS PEAK MEM USAGE:",
                   "DMEM = ", MYTDDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", MYTDDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))


        if self.verbose >= 2:
            _print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        return logbook
    #################################################

    
    #################################################
    def save_gs_mps(self, save_dir='./gs_mps'):
        import shutil
        import pickle
        import os
        if self.mpi is None or self.mpi.rank == 0:
            pickle.dump(self.gs_energy, open(
                self.scratch + '/GS_ENERGY', 'wb'))
            for k in os.listdir(self.scratch):
                if '.KET.' in k or k == 'GS_MPS_INFO' or k == 'GS_ENERGY':
                    shutil.copy(self.scratch + "/" + k, save_dir + "/" + k)
        if self.mpi is not None:
            self.mpi.barrier()
    #################################################

    
    #################################################
    def load_gs_mps(self, load_dir='./gs_mps'):
        import shutil
        import pickle
        import os
        if self.mpi is None or self.mpi.rank == 0:
            for k in os.listdir(load_dir):
                shutil.copy(load_dir + "/" + k, self.scratch + "/" + k)
        if self.mpi is not None:
            self.mpi.barrier()
        self.gs_energy = pickle.load(open(self.scratch + '/GS_ENERGY', 'rb'))
    #################################################


    #################################################
    def print_occupation_table(self, dm, aorb):

        from pyscf import symm
        
        natocc_a = eigvalsh(dm[0,:,:])
        natocc_a = natocc_a[::-1]
        natocc_b = eigvalsh(dm[1,:,:])
        natocc_b = natocc_b[::-1]

        mk = ''
        if isinstance(aorb, int):
            mk = ' (ann.)'
        ll = 4 + 16 + 15 + 12 + (0 if isinstance(aorb, int) else 13) + \
             (3+18) + 18 + 2*len(mk)
        hline = ''.join(['-' for i in range(0, ll)])
        aspace = ''.join([' ' for i in range(0,len(mk))])

        _print(hline)
        _print('%4s'  % 'No.', end='') 
        _print('%16s' % 'Alpha MO occ.' + aspace, end='')
        _print('%15s' % 'Beta MO occ.' + aspace,  end='')
        _print('%13s' % 'Irrep / ID',  end='')
        if isinstance(aorb, np.ndarray):
            _print('%13s' % 'aorb coeff', end='')
        _print('   %18s' % 'Alpha natorb occ.', end='')
        _print('%18s' % 'Beta natorb occ.',  end='\n')
        _print(hline)

        for i in range(0, dm.shape[1]):
            if isinstance(aorb, int):
                mk0 = aspace
                if i == aorb: mk0 = mk
            else:
                mk0 = aspace

            _print('%4d' % i, end='')
            _print('%16.8f' % np.diag(dm[0,:,:])[i].real + mk0, end='')
            _print('%15.8f' % np.diag(dm[1,:,:])[i].real + mk0, end='')
            j = i if self.ridx is None else self.ridx[i]
            sym_label = symm.irrep_id2name(self.groupname, self.orb_sym[j])
            _print('%13s' % (sym_label + ' / ' + str(self.orb_sym[j])), end='')
            if isinstance(aorb, np.ndarray):
                _print('%13.8f' % aorb[i], end='')
            _print('   %18.8f' % natocc_a[i], end='')
            _print('%18.8f' % natocc_b[i], end='\n')

        _print(hline)
        _print('%4s' % 'Sum', end='')
        _print('%16.8f' % np.trace(dm[0,:,:]).real + aspace, end='')
        _print('%15.8f' % np.trace(dm[1,:,:]).real + aspace, end='')
        _print('%13s' % ' ', end='')
        if isinstance(aorb, np.ndarray):
            _print('%13s' % ' ', end='')
        _print('   %18.8f' % sum(natocc_a), end='')
        _print('%18.8f' % sum(natocc_b), end='\n')
        _print(hline)
    #################################################

    
    #################################################
    def annihilate(self, logbook_in, aorb, fit_bond_dims, fit_noises, fit_conv_tol, fit_n_steps, 
                   pg, inmps_dir0=None, inmps_name='GS_MPS_INFO', outmps_dir0=None,
                   outmps_name='ANN_MPS_INFO', aorb_thr=1.0E-12, alpha=True, cutoff=1E-14, occs=None,
                   bias=1.0, outmps_normal=True, save_1pdm=False, out_singlet_embed=False, 
                   out_cpx=False, mrci_info=None, mps_tag=None):
        """
        aorb can be int, numpy.ndarray, or 'nat<n>' where n is an integer'
        """
        logbook = logbook_in.copy()
        ##OLD ops = [None] * len(aorb)
        ##OLD rkets = [None] * len(aorb)
        ##OLD rmpos = [None] * len(aorb)
        from pyscf import symm

        if self.mpi is not None:
            self.mpi.barrier()

        if inmps_dir0 is None:
            inmps_dir = self.scratch
        else:
            inmps_dir = inmps_dir0
        if outmps_dir0 is None:
            outmps_dir = self.scratch
        else:
            outmps_dir = outmps_dir0

            
        #==== Checking input parameters ====#
        if not (isinstance(aorb, int) or isinstance(aorb, np.ndarray) or
                isinstance(aorb, str)):
            raise ValueError('The argument \'aorb\' of MYTDDMRG.annihilate method must ' +
                             'be either an integer, a numpy.ndarray, or a string. ' +
                             f'Currently, aorb = {aorb}.')

        use_natorb = False
        if isinstance(aorb, str):
            nn = len(aorb)
            ss = aorb[0:3]
            ss_ = aorb[3:nn]
            if ss != 'nat' or not ss_.isdigit:
                _print('aorb = ', aorb)
                raise ValueError(
                    'The only way the argument \'aorb\' of MYTDDMRG.annihilate ' +
                    'can be a string is when it has the format of \'nat<n>\', ' +
                    'where \'n\' must be an integer, e.g. \'nat1\', \'nat24\', ' +
                    f'etc. Currently aorb = {aorb:s}')
            nat_id = int(ss_)
            if nat_id < 0 or nat_id >= self.n_sites:
                raise ValueError('The index of the natural orbital specified by ' +
                                 'the argument \'aorb\' of MYTDDMRG.annihilate ' +
                                 'is out of bound, which is between 0 and ' +
                                 f'{self.n_sites:d}. Currently, the specified index ' +
                                 f'is {nat_id:d}.')
            use_natorb = True

            
        idMPO_ = bs.SimplifiedMPO(bs.IdentityMPO(self.hamil), bs.RuleQC(), True, True)
        if self.mpi is not None:
            idMPO_ = bs.ParallelMPO(idMPO_, self.identrule)

                
        #==== Build Hamiltonian MPO (to provide noise ====#
        #====      for Linear.solve down below)       ====#
        if self.mpo_orig is None:
            mpo = bs.MPOQC(self.hamil, b2.QCTypes.Conventional)
            mpo = bs.SimplifiedMPO(mpo, bs.RuleQC(), True, True,
                                   b2.OpNamesSet((b2.OpNames.R, b2.OpNames.RD)))
            self.mpo_orig = mpo
        else:
            mpo = self.mpo_orig
        if self.mpi is not None:
            mpo = bs.ParallelMPO(mpo, self.prule)
        #debug print_MPO_bond_dims(mpo, 'Hamiltonian')  # the printed bonddim is different from MPO contruction step.


        #==== Load the input MPS ====#
        complex_mps = (comp == 'full')
        if mrci_info is not None:
            mps_type = {'type':'mrci', 'nactive2':mrci_info['nactive2'], 'order':mrci_info['order'],
                        'n_sites':self.n_sites, 'vacuum':self.hamil.vacuum, 'target':self.target,
                        'basis':self.hamil.basis}
        else:
            mps_type = {'type':'normal'}
        mps, mps_info, _ = \
                loadMPSfromDir(inmps_dir, inmps_name, complex_mps, mps_type, idMPO_,
                               cached_contraction=True, MPI=self.mpi, 
                               prule=self.prule if self.mpi is not None else None)  # 1)
        # At the moment, the annihilator function does not support input MPS of the
        # type multi MPS.
        _print('Input MPS max. bond dimension = ', mps.info.bond_dim)
        assert mps_info.target.n == self.nel_site, \
            'The number of active space electrons from the quantum number label does ' + \
            'not mathc the one specified in the input file.'


        #==== Compute the requested natural orbital ====#
        dm0 = self.get_one_pdm(comp=='full', mps)
        if use_natorb:
            e0, aorb = eigh(dm0[0 if alpha else 1,:,:])
            aorb = aorb[:,::-1]       # Reverse the columns
            aorb = aorb[:,nat_id]
            aorb[np.abs(aorb) < aorb_thr] = 0.0          # 1)
            _print(aorb)
        _print('Occupations before annihilation:')
        self.print_occupation_table(dm0, aorb)
        # NOTES:
        # 1) For some reason, without setting the small coefficients to zero,
        #    a segfault error happens later inside the MPS_fitting function.

        
        #==== Some statistics ====#
        if self.print_statistics:
            max_d = max(fit_bond_dims)
            mps_info2 = brs.MPSInfo(self.n_sites, self.hamil.vacuum,
                                    self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("EST MAX OUTPUT MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("EST PEAK MEM = ", MYTDDMRG.fmt_size(mem2),
                   " SCRATCH = ", MYTDDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        #NOTE: check if ridx is not none

        
        #==== Determine the reordered index of the annihilated orbital ====#
        if isinstance(aorb, int):
            idx = aorb if self.ridx is None else self.ridx[aorb]
            # idx is the index of the annihilated orbital after reordering.
        elif isinstance(aorb, np.ndarray):
            if self.idx is not None:
                aorb = aorb[self.idx]            
            
            #== Determine the irrep. of aorb ==#
            for i in range(0, self.n_sites):
                j = i if self.ridx is None else self.ridx[i]
                if (np.abs(aorb[j]) >= aorb_thr):
                    aorb_sym = self.orb_sym[j]            # 1)
                    logbook.update({'ann:osym':aorb_sym})
                    break
            for i in range(0, self.n_sites):
                j = i if self.ridx is None else self.ridx[i]
                if (np.abs(aorb[j]) >= aorb_thr and self.orb_sym[j] != aorb_sym):
                    _print(self.orb_sym)
                    _print('An inconsistency in the orbital symmetry found in aorb: ')
                    _print(f'   The first detected nonzero element has a symmetry ID of {aorb_sym:d}, ', end='')
                    _print(f'but the symmetry ID of another nonzero element (the {i:d}-th element) ', end='')
                    _print(f'is {self.orb_sym[j]:d}.')
                    raise ValueError('The orbitals making up the linear combination in ' +
                                     'aorb must all have the same symmetry.')
            # NOTES:
            # 1) j instead of i is used as the index for self.orb_sym because the
            #    contents of this array have been reordered (see its assignment in the
            #    self.init_hamiltonian or self.init_hamiltonian_fcidump function).

                
        #==== Begin constructing the annihilation MPO ====#
        gidxs = list(range(self.n_sites))
        if isinstance(aorb, int):
            if spin_symmetry == 'sz':
                ops = bs.OpElement(b2.OpNames.D,
                                   b2.SiteIndex((idx, ), (0 if alpha else 1, )), 
                                   SX(-1, -1 if alpha else 1, self.orb_sym[idx]))
            elif spin_symmetry == 'su2':
                ops = bs.OpElement(b2.OpNames.D, b2.SiteIndex((idx, ), ()), 
                                   SX(-1, 1, self.orb_sym[idx]))
        elif isinstance(aorb, np.ndarray):
            ops = [None] * self.n_sites
            for ii, ix in enumerate(gidxs):
                if spin_symmetry == 'sz':
                    ops[ii] = bs.OpElement(b2.OpNames.D,
                                           b2.SiteIndex((ix, ), (0 if alpha else 1, )),
                                           SX(-1, -1 if alpha else 1, aorb_sym))
                elif spin_symmetry == 'su2':
                    ops[ii] = bs.OpElement(b2.OpNames.D, b2.SiteIndex((ix, ), ()), 
                                           SX(-1, 1, aorb_sym))
                    

        #==== Determine if the annihilated orbital is a site orbital ====#
        #====  or a linear combination of them (orbital transform)   ====#
        if isinstance(aorb, int):
            rmpos = bs.SimplifiedMPO(
                bs.SiteMPO(self.hamil, ops), bs.NoTransposeRule(bs.RuleQC()), True, True,
                b2.OpNamesSet((b2.OpNames.R, b2.OpNames.RD)))
        elif isinstance(aorb, np.ndarray):
            ao_ops = bs.VectorOpElement([None] * self.n_sites)
            for ix in range(self.n_sites):
                ao_ops[ix] = ops[ix] * aorb[ix]
                _print('opsx = ', ops[ix], type(ops[ix]), ao_ops[ix], type(ao_ops[ix]))
            rmpos = bs.SimplifiedMPO(
                bs.LocalMPO(self.hamil, ao_ops), bs.NoTransposeRule(bs.RuleQC()), True, True,
                b2.OpNamesSet((b2.OpNames.R, b2.OpNames.RD)))
        if self.mpi is not None:
            rmpos = bs.ParallelMPO(rmpos, self.siterule)

                                
        if self.mpi is not None:
            self.mpi.barrier()
        if self.verbose >= 2:
            _print('>>> START : Applying the annihilation operator <<<')
        t = time.perf_counter()


        #==== Determine the quantum numbers of the output MPS, rkets ====#
        if isinstance(aorb, int):
            ion_target = self.target + ops.q_label
        elif isinstance(aorb, np.ndarray):
            ion_sym = self.wfn_sym ^ aorb_sym
            ion_target = SX(self.nel_site-1, 1, ion_sym)
            logbook.update({'ann:wsym':ion_sym})
        rket_info = brs.MPSInfo(self.n_sites, self.hamil.vacuum, ion_target,
                                self.hamil.basis)
        
        _print('Quantum number information:')
        _print(' - Input MPS = ', self.target)
        _print(' - Input MPS multiplicity = ', self.target.multiplicity)
        if isinstance(aorb, int):
            _print(' - Annihilated orbital = ', ops.q_label)
        elif isinstance(aorb, np.ndarray):
            _print(' - Annihilated orbital = ', SX(-1, 1, aorb_sym))
        _print(' - Output MPS = ', ion_target)
        _print(' - Output MPS multiplicity = ', ion_target.multiplicity)
        logbook.update({'ann:qnumber:n':ion_target.n,
                        'ann:qnumber:mult':ion_target.multiplicity,
                        'ann:qnumber:pg':ion_target.pg})


        #==== Tag the output MPS ====#
        if mps_tag is None:
            if isinstance(aorb, int):
                rket_info.tag = 'DKET_%d' % idx
            elif isinstance(aorb, np.ndarray):
                rket_info.tag = 'DKET_C'
        else:
            rket_info.tag = mps_tag
        logbook.update({'ann:tag':rket_info.tag})
        

        #==== Set the bond dimension of output MPS ====#
        rket_info.set_bond_dimension(mps.info.bond_dim)
        if occs is None:
            if self.verbose >= 2:
                _print("Using FCI INIT MPS")
            rket_info.set_bond_dimension(mps.info.bond_dim)
        else:
            if self.verbose >= 2:
                _print("Using occupation number INIT MPS")
            if self.idx is not None:
                occs = occs[self.idx]
            rket_info.set_bond_dimension_using_occ(
                mps.info.bond_dim, b2.VectorDouble(occs), bias=bias)


        #==== Initialization of output MPS ====#
        rket_info.save_data(self.scratch + "/" + outmps_name)
        rkets = bs.MPS(self.n_sites, mps.center, 2)
        rkets.initialize(rket_info)
        rkets.random_canonicalize()
        rkets.save_mutable()
        rkets.deallocate()
        rket_info.save_mutable()
        rket_info.deallocate_mutable()

        #OLD if mo_coeff is None:
        #OLD     # the mpo and gf are in the same basis
        #OLD     # the mpo is SiteMPO
        #OLD     rmpos = SimplifiedMPO(
        #OLD         SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        #OLD else:
        #OLD     # the mpo is in mo basis and gf is in ao basis
        #OLD     # the mpo is sum of SiteMPO (LocalMPO)
        #OLD     ao_ops = VectorOpElement([None] * self.n_sites)
        #OLD     _print('mo_coeff = ', mo_coeff)
        #OLD     for ix in range(self.n_sites):
        #OLD         ao_ops[ix] = ops[ix] * mo_coeff[ix]
        #OLD         _print('opsx = ', ops[ix], type(ops[ix]), ao_ops[ix], type(ao_ops[ix]))
        #OLD     rmpos = SimplifiedMPO(
        #OLD         LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        #OLD     
        #OLD if self.mpi is not None:
        #OLD     rmpos = ParallelMPO(rmpos, self.siterule)


        #==== Solve for the output MPS ====#
        MPS_fitting(rkets, mps, rmpos, fit_bond_dims, fit_n_steps, fit_noises,
                    fit_conv_tol, 'density_mat', cutoff, lmpo=mpo,
                    verbose_lvl=self.verbose-1)
        _print('Output MPS max. bond dimension = ', rkets.info.bond_dim)

            
        #==== Normalize the output MPS if requested ====#
        if outmps_normal:
            _print('Normalizing the output MPS')
            icent = rkets.center
            #OLD if rkets.dot == 2 and rkets.center == rkets.n_sites-1:
            if rkets.dot == 2:
                if rkets.center == rkets.n_sites-2:
                    icent += 1
                elif rkets.center == 0:
                    pass
            assert rkets.tensors[icent] is not None
            
            rkets.load_tensor(icent)
            rkets.tensors[icent].normalize()
            rkets.save_tensor(icent)
            rkets.unload_tensor(icent)
            # rket_info.save_data(self.scratch + "/" + outmps_name)


        #==== Check the norm ====#
        idMPO_ = bs.SimplifiedMPO(bs.IdentityMPO(self.hamil), bs.RuleQC(), True, True)
        if self.mpi is not None:
            idMPO_ = bs.ParallelMPO(idMPO_, self.identrule)
        idN = bs.MovingEnvironment(idMPO_, rkets, rkets, "norm")
        idN.init_environments()
        nrm = bs.Expect(idN, rkets.info.bond_dim, rkets.info.bond_dim)
        nrm_ = nrm.solve(False)
        _print('Output MPS norm = ',
               '%11.8f, %11.8fj' % (nrm_.real, nrm_.imag) if comp=='full' else
               '%11.8f' % nrm_)
        logbook.update({'ann:norm':nrm_})

            
        #==== Print the energy of the output MPS ====#
        energy = calc_energy_MPS(mpo, rkets, 0)
        _print('Output MPS energy = ',
               '(%12.8f, %12.8fj) Hartree' % (energy.real, energy.imag) if comp=='full' else
               '%12.8f Hartree' % energy)
        _print('Canonical form of the annihilation output (ortho. center) = ' +
               f'{rkets.canonical_form} ({rkets.center})')
        logbook.update({'ann:energy':energy, 'ann:canonical_form':rkets.canonical_form,
                        'ann:center':rkets.center})
        dm1 = self.get_one_pdm(comp=='full', rkets)
        _print('Occupations after annihilation:')
        if isinstance(aorb, int):
            self.print_occupation_table(dm1, aorb)
        elif isinstance(aorb, np.ndarray):
            if self.ridx is None:
                self.print_occupation_table(dm1, aorb)
            else:
                self.print_occupation_table(dm1, aorb[self.ridx])

           
        #==== Partial charge ====#
        dm1_full = make_full_dm(self.n_core, dm1)
        orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()), axis=2)
        self.qmul1, self.qlow1 = \
            pcharge.calc(self.mol, dm1_full, orbs, self.ovl_ao)
        print_pcharge(self.mol, self.qmul1, self.qlow1)
        logbook.update({'ann:mulliken':self.qmul1, 'ann:lowdin':self.qlow1})

        #==== Bond order ====#
        self.bo_mul1, self.bo_low1 = bond_order.calc(self.mol, dm1_full, orbs, self.ovl_ao)
        print_section('Mulliken bond orders', 2)
        print_bond_order(self.bo_mul1)
        print_section('Lowdin bond orders', 2)
        print_bond_order(self.bo_low1)
        
        #==== Multipole analysis ====#
        e_dpole, n_dpole, e_qpole, n_qpole = \
            mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm1_full, orbs)
        print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)
        logbook.update({'ann:e_dipole':e_dpole, 'ann:n_dipole':n_dpole,
                        'ann:e_quadpole':e_qpole, 'ann:n_quadpole':n_qpole})


        #==== Singlet embedding ====#
        if out_singlet_embed:
            init_target, init_multip = rkets.info.target, rkets.info.target.multiplicity
            init_form, init_center = rkets.canonical_form, rkets.center
            rkets = trans_to_singlet_embed(rkets, rkets.info.tag, self.prule)
            rkets.save_data()
            rkets.info.save_data(self.scratch + "/" + outmps_name)
            _print('')
            _print('The output MPS is transformed to a singlet embedded MPS.')
            _print('Quantum number information with singlet embedding:')
            _print(' - Input MPS = ', init_target)
            _print(' - Input MPS multiplicity = ', init_multip)
            _print(' - Input canonical form (ortho. center) = ' +
               f'{init_form} ({init_center})')
            _print(' - Output MPS = ', rkets.info.target)
            _print(' - Output MPS multiplicity = ', rkets.info.target.multiplicity)
            _print(' - Output canonical form (ortho. center) = ' +
               f'{rkets.canonical_form} ({rkets.center})')
            logbook.update({'ann:qnumber:n':rkets.info.target.n,
                            'ann:qnumber:mult':rkets.info.target.multiplicity,
                            'ann:qnumber:pg':rkets.info.target.pg,
                            'ann:canonical_form':rkets.canonical_form,
                            'ann:center':rkets.center})

            
        #==== Conversion to full complex MPS ====#
        if out_cpx:
            assert comp != 'full'
            _print('Converting the converged MPS to a complex MPS ...')
            rkets = self.b2driver.mps_change_complex(rkets, "CPX")
            
        #==== Save the output MPS ====#
        _print('')
        if outmps_dir != self.scratch:
            b2tools.mkDir(outmps_dir)
        rkets.info.save_data(outmps_dir + "/" + outmps_name)
        _print('Saving output MPS files under ' + outmps_dir)
        saveMPStoDir(rkets, outmps_dir, self.mpi)
        if save_1pdm:
            _print('Saving 1PDM of the output MPS under ' + outmps_dir)
            np.save(outmps_dir + '/ANN_1pdm', dm1)

            
        if self.verbose >= 2:
            _print('>>> COMPLETE : Application of annihilation operator | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        return logbook
    #################################################


    #################################################
    def save_time_info(self, save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps,
                       save_1pdm, rs, re, ro, dm):

        def save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps,
                            save_1pdm, rs, re, ro, dm):
            yn_bools = ('No','Yes')
            au2fs = 2.4188843265e-2   # a.u. of time to fs conversion factor
            with open(save_dir + '/TIME_INFO', 'w') as t_info:
                t_info.write(' Reasons printed = \n')
                if rs: t_info.write('    * sampling time \n')
                if re: t_info.write('    * last time point \n')
                if ro: t_info.write('    * probe file \n')
                t_info.write(' Actual sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
                             (it, t, t*au2fs))
                t_info.write(' Requested sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
                             (i_sp, t_sp, t_sp*au2fs))
                t_info.write(' MPS norm square = %19.14f\n' % normsq)
                t_info.write(' Autocorrelation = (%19.14f, %19.14f)\n' % (ac.real, ac.imag))
                t_info.write(f' Is MPS saved at this time?  {yn_bools[save_mps]}\n')
                t_info.write(f' Is 1PDM saved at this time?  {yn_bools[save_1pdm]}\n')
                if save_1pdm:
                    natocc_a = eigvalsh(dm[0,:,:])
                    natocc_b = eigvalsh(dm[1,:,:])
                    t_info.write(' 1PDM info:\n')
                    t_info.write('    Trace (alpha,beta) = (%16.12f,%16.12f) \n' %
                                 ( np.trace(dm[0,:,:]).real, np.trace(dm[1,:,:]).real ))
                    t_info.write('    ')
                    for i in range(0, 4+4*20): t_info.write('-')
                    t_info.write('\n')
                    t_info.write('    ' +
                                 '%4s'  % 'No.' + 
                                 '%20s' % 'Alpha MO occ.' +
                                 '%20s' % 'Beta MO occ.' +
                                 '%20s' % 'Alpha natorb occ.' +
                                 '%20s' % 'Beta natorb occ.' + '\n')
                    t_info.write('    ')
                    for i in range(0, 4+4*20): t_info.write('-')
                    t_info.write('\n')
                    for i in range(0, dm.shape[1]):
                        t_info.write('    ' +
                                     '%4d'  % i + 
                                     '%20.12f' % np.diag(dm[0,:,:])[i].real +
                                     '%20.12f' % np.diag(dm[1,:,:])[i].real +
                                     '%20.12f' % natocc_a[i].real +
                                     '%20.12f' % natocc_b[i].real + '\n')
                
        if self.mpi is not None:
            if self.mpi.rank == 0:
                save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps, save_1pdm,
                                rs, re, ro, dm)
            self.mpi.barrier()
        else:
            save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps, save_1pdm,
                            rs, re, ro, dm)
    #################################################


    #################################################
    def save_time_info_ow(self, dir_ow, t, it, t_sp, i_sp, normsq, ac):

        def save_time_info_ow0(dir_ow, t, it, t_sp, i_sp, normsq, ac):
            yn_bools = ('No','Yes')
            au2fs = 2.4188843265e-2   # a.u. of time to fs conversion factor
            with open(dir_ow + '/TIME_INFO', 'w') as t_info:
                t_info.write(' Actual sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
                             (it, t, t*au2fs))
                t_info.write(' Requested sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
                             (i_sp, t_sp, t_sp*au2fs))
                t_info.write(' MPS norm square = %19.14f\n' % normsq)
                t_info.write(' Autocorrelation = (%19.14f, %19.14f)\n' % (ac.real, ac.imag))
                            
        if self.mpi is not None:
            if self.mpi.rank == 0:
                save_time_info_ow0(dir_ow, t, it, t_sp, i_sp, normsq, ac)
            self.mpi.barrier()
        else:
            save_time_info_ow0(dir_ow, t, it, t_sp, i_sp, normsq, ac)
    #################################################


    #################################################
    def get_te_times(self, dt0, tmax, tinit=0.0):

        #OLD n_steps = int(tmax/dt + 1)
        #OLD ts = np.linspace(0, tmax, n_steps)    # times
        if type(dt0) is not list:
            dt = [dt0]
        else:
            dt = dt0
        ts = [tinit]
        i = 1
        while ts[-1] < tmax:
            if i <= len(dt):
                ts = ts + [tinit + sum(dt[0:i])]
            else:
                ts = ts + [ts[-1] + dt[-1]]
            i += 1
        if ts[-1] > tmax:
            ts[-1] = tmax
            if abs(ts[-1]-ts[-2]) < 1E-3:
                ts.pop()
                ts[-1] = tmax
        ts = np.array(ts)
        return ts
    #################################################


    #################################################
    def read_probe_file(self, drt, fc):
        #==== Get the list of files in drt ====#
        fl = glob.glob(drt + '/probe-*')
        prefixlen = len('probe-')
        goodfile, save_mps, save_1pdm = False, False, False

        #==== Return when no probe file is found ====#
        if len(fl) == 0:
            return False, False, False

        readit = False
        for f in fl:
            #==== Strip the directory name ====#
            fn = os.path.basename(f)

            #==== Determine if the probe file for the fc-th step exists ====#
            if fn[prefixlen:].isdigit():
                fc0 = int(fn[prefixlen:])
                if fc0 == fc:
                    fn_full = f
                    readit = True
                    break

        truel =  ('true',  't', '1', 'yes', 'y')
        falsel = ('false', 'f', '0',  'no', 'n')

        #==== Read the probe file for the fc-th step ====#
        if readit:
            with open(fn_full, 'r') as infile:
                lines_ = infile.read()
                lines = lines_.split('\n')
                goodfile, save_mps, save_1pdm = False, False, False

                #== Parse the content of the probe file ==#
                for l in lines:
                    w = l.split()
                    if len(w) == 2 and w[0].lower() == 'save_mps':
                        if w[1].lower() in truel:
                            save_mps = True
                        elif w[1].lower() in falsel:
                            save_mps = False
                    elif len(w) == 2 and w[0].lower() == 'save_1pdm':
                        if w[1].lower() in truel:
                            save_1pdm = True
                        elif w[1].lower() in falsel:
                            save_1pdm = False
                    elif len(w) == 0:
                        pass       # blank lines are tolerated.
                    else:
                        print_warning(f'A probe file {fn_full} is found but will be ignored ' + \
                                      'because it does not follow the correct format.')
                        return False, False, False
                goodfile = True
                print_section('Probe file')
                _print('  A format-conforming probe file:')
                _print(f'      {fn_full}')
                _print('  is found requesting to save:')
                if save_mps: _print( '     * MPS')
                if save_1pdm: _print('     * one-particle RDM')
        else:
            return False, False, False
        
        return goodfile, save_mps, save_1pdm
    #################################################


    ##################################################################
    def time_propagate(self, logbook, max_bond_dim: int, method, tmax: float, dt0: float, 
                       tinit=0.0, inmps_dir0=None, inmps_name='ANN_MPS_INFO', inmps_cpx=False,
                       inmps_multi=False, mps_act0_dir=None, mps_act0_name=None,
                       mps_act0_cpx=None, mps_act0_multi=None, exp_tol=1e-6, cutoff=0, 
                       normalize=False, n_sub_sweeps=2, n_sub_sweeps_init=4, 
                       krylov_size=20, krylov_tol=5.0E-6, t_sample0=None, 
                       save_mps='overwrite', save_1pdm=False, save_2pdm=False, prefix='te', 
                       save_txt=True, save_npy=False, in_singlet_embed=False, 
                       se_nel_site=None, mrci_info=None, bo_pairs=None, prefit=False, 
                       prefit_bond_dims=None, prefit_nsteps=None, prefit_noises=None, 
                       prefit_conv_tol=None, prefit_cutoff=None, verbosity=6):
        '''
        Coming soon
        '''

        #OLD_CPX if self.mpi is not None:
        #OLD_CPX     if SpinLabel == SU2:
        #OLD_CPX         from block2.su2 import ParallelMPO
        #OLD_CPX     else:
        #OLD_CPX         from block2.sz import ParallelMPO

            
        #==== Needed directories ====#
        if inmps_dir0 is None:
            inmps_dir = self.scratch
        else:
            inmps_dir = inmps_dir0
        assert save_mps=='overwrite' or save_mps=='sampled' or save_mps=='no'
        sample_dir = logbook['workdir'] + '/' + prefix + '.sample'
        mps_dir_ow = logbook['workdir'] + '/' + prefix + '.mps_t'
        logbook.update({'sample_dir':sample_dir})
        if save_mps == 'overwrite': logbook.update({'mps_dir_ow':mps_dir_ow})
        if self.mpi is not None:
            if self.mpi.rank == 0:
                if save_mps == 'overwrite': b2tools.mkDir(mps_dir_ow)
                self.mpi.barrier()
        else:
            if save_mps == 'overwrite': b2tools.mkDir(mps_dir_ow)
            

        if comp == 'full':
            assert inmps_cpx, "When complex type is 'full', inmps_cpx must be true, " + \
                'that is, the initial MPS must be complex.'

            
        #==== Construct propagation time vector ====#
        ts = self.get_te_times(dt0, tmax, tinit)
        n_steps = len(ts)
        ndigit = len(str(n_steps))
        _print('Time points (a.u.) = ', ts)

        #==== Construct sampling time vector ====#
        if t_sample0 is not None:
            if isinstance(t_sample0, np.ndarray):
                t_sample = list(t_sample0)
            elif isinstance(t_sample0, (list, tuple)):
                if t_sample0[0] == 'steps':
                    assert len(t_sample0) == 2 and isinstance(t_sample0[1], int)
                    t_sample = list( ts[0::t_sample0[1]] )
                elif t_sample0[0] == 'delta':
                    assert len(t_sample0) == 2 and isinstance(t_sample0[1], float)
                    dt_s = t_sample0[1]
                    t_sample = list( np.linspace(tinit, round(tmax/dt_s)*dt_s,
                                                 round((tmax-tinit)/dt_s)+1) )
                else:
                    raise ValueError('When t_sample is a length 2 list or tuple, ' +
                                     'the first element must be either \'steps\' ' +
                                     'or \'delta\'.')
            else:
                raise ValueError('The value of t_sample0 is non-conforming to the ' +
                                 'available options.')

            if abs(t_sample[0]-tinit) > 1.0E-12:
                t_sample = [tinit] + t_sample
            if abs(t_sample[-1]-tmax) > 1.0E-12:
                t_sample = t_sample + [tmax]
            _print('Sampling time points (a.u.) = ', t_sample)
            
        #==== Initiate autocorrelations file ====#
        ac_print = print_autocorrelation(prefix, len(ts), save_txt, save_npy)
        if self.mpi is None or self.mpi.rank == 0:
            ac_print.header()
        
        #==== Initiate Lowdin partial charges file ====#
        if t_sample is not None:
            atom_symbol = [self.mol.atom_symbol(i) for i in range(0, self.mol.natm)]
            q_print = print_td_pcharge(atom_symbol, prefix, len(t_sample), 8, save_txt,
                                       save_npy)
            if self.mpi is None or self.mpi.rank == 0:
                q_print.header()
            if self.mpi is not None: self.mpi.barrier()

        #==== Initiate Lowdin bond order file ====#
        if t_sample is not None and bo_pairs is not None:
            bo_print = print_td_bo(bo_pairs, atom_symbol, prefix, len(t_sample), 8, 
                                   save_txt, save_npy)
            if self.mpi is None or self.mpi.rank == 0:
                bo_print.header()
            if self.mpi is not None: self.mpi.barrier()

        #==== Initiate multipole components file ====#
        if t_sample is not None:
            mp_print = print_td_mpole(prefix, len(t_sample), save_txt, save_npy)
            if self.mpi is None or self.mpi.rank == 0:
                mp_print.header()
            if self.mpi is not None: self.mpi.barrier()
                
        #==== Prepare Hamiltonian MPO ====#
        if self.mpi is not None:
            self.mpi.barrier()
        #if self.mpo_orig is None:
        #    mpo = bs.MPOQC(self.hamil, b2.QCTypes.Conventional)
        #    mpo = bs.SimplifiedMPO(mpo, bs.RuleQC(), True, True,
        #                           b2.OpNamesSet((b2.OpNames.R, b2.OpNames.RD)))
        #    self.mpo_orig = mpo
        #else:
        #    mpo = self.mpo_orig
        ##need? mpo = IdentityAddedMPO(mpo) # hrl: alternative
        #if self.mpi is not None:
        #    mpo = bs.ParallelMPO(mpo, self.prule)
        #debug print_MPO_bond_dims(self.te_mpo, 'Hamiltonian')   # the printed bonddim is different from MPO contruction step.

        
        #==== Load the initial MPS ====#
        loadv2 = True
        if loadv2:
            idMPO = bs.SimplifiedMPO(bs.IdentityMPO(self.hamil), bs.RuleQC(), True, True)
            if self.mpi is not None:
                idMPO = bs.ParallelMPO(idMPO, self.identrule)

            #==== Determine initial MPS type (normal, multi, or MRCI) ====#
            mps_type = {}
            if mrci_info is not None:
                assert not inmps_multi, 'At the moment MRCI MPS cannot be in multi MPS form.'
                mps_type.update({'type':'mrci', 'nactive2':mrci_info['nactive2'], 
                                 'order':mrci_info['order'], 'n_sites':self.n_sites,
                                 'vacuum':self.hamil.vacuum, 'target':self.target,
                                 'basis':self.hamil.basis})
            else:
                if inmps_multi:
                    mps_type['type'] = 'multi'
                    mps_type['nroots'] = 2
                else:
                    mps_type['type'] = 'normal'

            _print('Loading initial MPS info from ' + inmps_dir + "/" + inmps_name)
            mps, mps_info, _ = \
                loadMPSfromDir(inmps_dir, inmps_name, inmps_cpx, mps_type, idMPO,
                               cached_contraction=True, MPI=self.mpi, 
                               prule=self.prule if self.mpi is not None else None)

            
            #==== Determine the type of t=0 MPS for autocorrelation ====#
            #====             (normal, multi, or MRCI)              ====#
            if mps_act0_dir is None: mps_act0_dir = inmps_dir
            if mps_act0_name is None: mps_act0_name = inmps_name
            if mps_act0_cpx is None: mps_act0_cpx = inmps_cpx
            if mps_act0_multi is None: mps_act0_multi = inmps_multi

            mps_act0_type = {}
            if mrci_info is not None:
                assert not mps_act0_multi, 'At the moment MRCI MPS cannot be in multi MPS form.'
                mps_act0_type.update(
                    {'type':'mrci', 'nactive2':mrci_info['nactive2'], 
                     'order':mrci_info['order'], 'n_sites':self.n_sites,
                     'vacuum':self.hamil.vacuum, 'target':self.target,
                     'basis':self.hamil.basis})
            else:
                if mps_act0_multi:
                    mps_act0_type['type'] = 'multi'
                    mps_act0_type['nroots'] = 2
                else:
                    mps_act0_type['type'] = 'normal'

            _print('Loading t=0 MPS for autocorrelation info from ' + mps_act0_dir + 
                   "/" + mps_act0_name)
            mps_act0, _, _ = \
                loadMPSfromDir(mps_act0_dir, mps_act0_name, mps_act0_cpx, 
                               mps_act0_type, idMPO, cached_contraction=True, 
                               MPI=self.mpi, prule=self.prule if self.mpi is
                               not None else None)
            #ipsh('After loading mps')
        else:
            raise NotImplementedError('Use loadv2.')
            inmps_path = inmps_dir + "/" + inmps_name
            mps_info = brs.MPSInfo(0)
            mps_info.load_data(inmps_path)
            mps = loadMPSfromDir_OLD(mps_info, inmps_dir, self.mpi)

            
        #==== Singlet embedding ====#
        if in_singlet_embed:
            _print('The input MPS is a singlet embedded MPS.')
            nel_t0 = se_nel_site + self.nel_core
        else:
            nel_t0 = self.nel_site + self.nel_core
        _print('Quantum number information:')
        _print(' - Initial MPS = ', mps_info.target)
        _print(' - Initial MPS multiplicity = ', mps_info.target.multiplicity)
        
        
        #==== Initial norm ====#
        idMPO = bs.SimplifiedMPO(bs.IdentityMPO(self.hamil), bs.RuleQC(), True, True)
        print_MPO_bond_dims(idMPO, 'Identity_2')
        if self.mpi is not None:
            idMPO = bs.ParallelMPO(idMPO, self.identrule)
        mps_n = mps.deep_copy('mps_norm')                 # 3)
        idN = bs.MovingEnvironment(idMPO, mps_n, mps_n, "norm_in")
        idN.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
        if inmps_cpx and inmps_multi:
            nrm = bs.ComplexExpect(idN, mps_n.info.bond_dim, mps_n.info.bond_dim)
        else:
            nrm = bs.Expect(idN, mps_n.info.bond_dim, mps_n.info.bond_dim)
        nrm_ = nrm.solve(False)
        _print(f'Initial MPS norm = Re: {nrm_.real:11.8f}, Im: {nrm_.imag:11.8f}')
        # 3) We duplicate mps here to a new identical mps_n rather than using the former
        #    because the norm calculation above changed the properties of mps such that
        #    the overlap with initial mps (mps_act0 below) is zero in the beginning, which
        #    should have been unity.

        
        #==== If a change of bond dimension of the initial MPS is requested ====#
        if prefit:
            if self.mpi is not None: self.mpi.barrier()
            ref_mps = mps.deep_copy('ref_mps_t0')
            if self.mpi is not None: self.mpi.barrier()
            MPS_fitting(mps, ref_mps, idMPO, prefit_bond_dims, prefit_nsteps, prefit_noises,
                        prefit_conv_tol, 'density_mat', prefit_cutoff, lmpo=idMPO,
                        verbose_lvl=self.verbose-1)


        #==== Make the input MPS complex when using hybrid complex ====#
        if inmps_cpx:
            # Just duplicate the input MPS if it is complex, regardless of
            # whether it is of multi MPS or normal MPS type. The comp type
            # does not matter either here.
            cmps = mps.deep_copy('mps_t')
        else:
            # If the input MPS is real (impliying comp=False), then use a
            # multi MPS to transform it to a complex multi MPS.
            cmps = bs.MultiMPS.make_complex(mps, "mps_t")
        _print('Initial canonical form (ortho. center) = ' +
               f'{cmps.canonical_form} ({cmps.center})')

        #==== Make the MPS for autocorrelation's t0 ====#
        #====   complex when using hybrid complex   ====#
        if mps_act0_cpx:
            cmps_act0 = mps_act0.deep_copy('mps_act0')
        else:
            cmps_act0 = bs.MultiMPS.make_complex(mps_act0, "mps_act0")


        #==== Take care of the algorithm type (1- or 2- site) ====#
        if mps.dot != 1: # change to 2dot
            cmps.load_data()
            cmps_act0.load_data()
            if comp == 'hybrid':
                cmps.canonical_form = 'M' + cmps.canonical_form[1:]
                cmps_act0.canonical_form = 'M' + cmps_act0.canonical_form[1:]
                print('mps canform = ', cmps.canonical_form)
                print('mps_t0 canform = ', cmps_act0.canonical_form)
            cmps.dot = 2
            cmps_act0.dot = 2
            cmps.save_data()
            cmps_act0.save_data()
            #ipsh('After checking dot')
        if cmps.dot == 2:
            _print('Algorithm type = 2-site')
        elif cmps.dot == 1:
            _print('Algorithm type = 1-site')


        #==== Initial setups for autocorrelation ====#
        idME = bs.MovingEnvironment(idMPO, cmps_act0, cmps, "acorr")
            
        
        #==== Initial setups for time evolution ====#
        #ipsh()
        #me = bs.MovingEnvironment(mpo, cmps, cmps, "TE")
        me = bs.MovingEnvironment(self.te_mpo, cmps, cmps, "TE")
        #ipsh()
        self.delayed_contraction = True
        if self.delayed_contraction:
            me.delayed_contraction = b2.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(self.verbose >= 2)


        #==== Time evolution ====#
        _print('Bond dim in TE : ', mps.info.bond_dim, max_bond_dim)
        if mps.info.bond_dim > max_bond_dim:
            _print('!!! WARNING !!!')
            _print('   The specified max. bond dimension for the time-evolved MPS ' +
                   f'({max_bond_dim:d}) is smaller than the max. bond dimension \n' +
                   f'  of the initial MPS ({mps.info.bond_dim:d}). This is in general not ' +
                   'recommended since the time evolution will always excite \n' +
                   '  correlation effects that are absent in the initial MPS.')
        if method == b2.TETypes.TangentSpace:
            te = bs.TimeEvolution(me, b2.VectorUBond([max_bond_dim]), method)
            te.krylov_subspace_size = krylov_size
            te.krylov_conv_thrd = krylov_tol
        elif method == b2.TETypes.RK4:
            te = bs.TimeEvolution(me, b2.VectorUBond([max_bond_dim]), method,
                                  n_sub_sweeps_init)
        elif method == b2.TETypes.RK4PP:
            te = bs.TDDMRG(me, b2.VectorUBond([max_bond_dim]))
            te.n_sub_sweeps = n_sub_sweeps_init
        te.cutoff = cutoff                    # for tiny systems, this is important
        te.iprint = verbosity
        if method != b2.TETypes.RK4PP:
            te.normalize_mps = normalize
        te.hermitian = True       # bcause CPX
        

        #==== Begin the time evolution ====#
        t_begin_tevo = time.time()
        logbook.update({'t_begin_tevo':t_begin_tevo})
        _print('Overhead time = ' + format_timespan(t_begin_tevo-logbook['t_start'],
                                                    max_units=5))
        
        if t_sample is not None:
            issampled = [False] * len(t_sample)
        if save_npy:
            np.save('./' + prefix + '.t', ts)
            if t_sample is not None: np.save('./'+prefix+'.ts', t_sample)
        i_sp = 0
        #ipsh()
        for it, tt in enumerate(ts):

            if self.verbose >= 2:
                _print('\n')
                _print(' Time point : ', it)
                _print('>>> TD-PROPAGATION TIME = %10.5f <<<' %tt)
            t = time.perf_counter()

            if it != 0: # time zero: no propagation
                dt_ = ts[it] - ts[it-1]
                _print('    DELTA_T stepped from the previous time point = %10.5f <<<' % dt_)
                if method == b2.TETypes.RK4:
                    te.solve(1, +1j * dt_, cmps.center == 0, tol=exp_tol)
                    te.n_sub_sweeps = n_sub_sweeps
                elif method == b2.TETypes.TangentSpace:
                    te.solve(2, +1j * dt_ / 2, cmps.center == 0, tol=exp_tol)
                    te.n_sub_sweeps = 1
                elif method == b2.TETypes.RK4PP:
                    _print('WARNING: TSTPP method is invoked. This method is not well-tested.')
                    te.solve(1, dt_, cmps.center == 0, tol=exp_tol)
                    te.n_sub_sweeps = n_sub_sweeps

                #if te.normalize_mps:
                #    _print('Normalizing the evolving MPS')
                #    icent = te.me.ket.center
                #    #OLD if te.me.ket.dot == 2 and te.me.ket.center == te.me.ket.n_sites-1:
                #    if te.me.ket.dot == 2:
                #        if te.me.ket.center == te.me.ket.n_sites-2:
                #            icent += 1
                #        elif te.me.ket.center == 0:
                #            pass
                #    assert te.me.ket.tensors[icent] is not None
                #    
                #    te.me.ket.load_tensor(icent)
                #    te.me.ket.tensors[icent].normalize()
                #    te.me.ket.save_tensor(icent)
                #    te.me.ket.unload_tensor(icent)
                     

                if comp == 'full':
                    _print(("T = %10.5f <E> = (%20.15f, %20.15f) <Norm^2> = %20.15f") %
                           (tt, te.energies[-1].real, te.energies[-1].imag, te.normsqs[-1]))
                else:
                    _print(("T = %10.5f <E> = %20.15f <Norm^2> = %20.15f") %
                           (tt, te.energies[-1], te.normsqs[-1]))
            else:
                _print('This is the starting time point, nothing happened yet.')
                        
            #==== Autocorrelation and norm ====#
            idME.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
            if comp == 'hybrid':
                acorr = brs.ComplexExpect(idME, max_bond_dim, max_bond_dim)
            elif comp == 'full':
                acorr = bs.Expect(idME, max_bond_dim, max_bond_dim)

            #acorr_t = 1.0
            acorr_t = acorr.solve(False)
            if it == 0:
                normsqs = nrm_.real    # abs(acorr_t)
            elif it > 0:
                normsqs = te.normsqs[0]
            acorr_t = acorr_t / np.sqrt(normsqs)
                            
            #==== 2t autocorrelation ====#
            if comp == 'hybrid':
                if cmps.wfns[0].data.size == 0:
                    loaded = True
                    cmps.load_tensor(cmps.center)
                vec = cmps.wfns[0].data + 1j * cmps.wfns[1].data
                acorr_2t = np.vdot(vec.conj(),vec) / normsqs
            elif comp == 'full':
                acorr_2t = complex(0.0, 0.0)

            #==== Print autocorrelation ====#
            if self.mpi is None or self.mpi.rank == 0:
                ac_print.print_ac(tt, acorr_t, acorr_2t, normsqs)

            if self.mpi is not None: self.mpi.barrier()

            
            #==== Determine reasons to save quantities ====#
            if t_sample is not None and it <= n_steps-2:
                dt1 = abs( ts[it]   - t_sample[i_sp] )
                dt2 = abs( ts[it+1] - t_sample[i_sp] )
                dd = (dt1 < dt2)
                r_sample = (dd and not issampled[i_sp])
            else:
                r_sample = False
            r_end = (it == n_steps-1)
            save_mps_end, save_1pdm_end = r_end, r_end
            r_probe, save_mps_probe, save_1pdm_probe = self.read_probe_file(sample_dir, it+1)

            
            #==== Compute and prob. store save quantities at sampling times ====#
            # 1) through t_sample,
            # 2) at the last time point, and
            # 3) requested by a probe file.
            if r_sample or r_end or r_probe:                    
                save_dir = sample_dir + '/tevo-' + str(it+1).zfill(ndigit)
                if self.mpi is not None:
                    if self.mpi.rank == 0:
                        b2tools.mkDir(save_dir)
                        self.mpi.barrier()
                else:
                    b2tools.mkDir(save_dir)

                #==== Saving MPS ====#
                if save_mps_end:
                    if save_mps == 'overwrite':
                        saveMPStoDir(cmps, mps_dir_ow, self.mpi)
                    elif save_mps == 'sampled':
                        saveMPStoDir(cmps, save_dir, self.mpi)
                    elif save_mps == 'no':
                        pass
                else:
                    if save_mps == 'overwrite':
                        saveMPStoDir(cmps, mps_dir_ow, self.mpi)
                    if save_mps == 'sampled' or save_mps_probe:
                        saveMPStoDir(cmps, save_dir, self.mpi)

                #==== Calculate 1PDM ====#
                if self.mpi is not None: self.mpi.barrier()
                cmps_cp = cmps.deep_copy('cmps_cp')         # 1)
                if self.mpi is not None: self.mpi.barrier()

                dm = self.get_one_pdm(True, cmps_cp)
                cmps_cp.info.deallocate()
                dm_full = make_full_dm(self.n_core, dm)
                dm_tr = np.sum( np.trace(dm_full, axis1=1, axis2=2) )
                dm_full = dm_full * nel_t0 / np.abs(dm_tr)      # dm_full is now normalized
                #OLD cmps_cp.deallocate()      # Unnecessary because it must have already been called inside the expect.solve function in the get_one_pdm above
                # NOTE:
                # 1) Copy the current MPS because self.get_one_pdm convert the input
                #    MPS to a real MPS.

                #==== Save 1PDM ====#
                if save_1pdm or save_1pdm_probe or save_1pdm_end:
                    np.save(save_dir+'/1pdm', dm)
                    
                #==== Save time info ====#
                if r_end:
                    sampled_mps_saved = save_mps_end if save_mps=='sampled' else False
                else:
                    sampled_mps_saved = save_mps=='sampled' or save_mps_probe
                self.save_time_info(save_dir, ts[it], it, t_sample[i_sp], i_sp, normsqs, 
                                    acorr_t, sampled_mps_saved,
                                    save_1pdm or save_1pdm_probe or save_1pdm_end,
                                    r_sample, r_end, r_probe, dm)
                if save_mps == 'overwrite':
                    self.save_time_info_ow(mps_dir_ow, ts[it], it, t_sample[i_sp], i_sp, 
                                           normsqs, acorr_t)
                                    

                if r_sample:
                    #==== Partial charges ====#
                    orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()),
                                          axis=2)
                    qmul, qlow = pcharge.calc(self.mol, dm_full, orbs, self.ovl_ao)
                    if self.mpi is None or self.mpi.rank == 0:
                        q_print.print_pcharge(tt, qlow)
                    if self.mpi is not None: self.mpi.barrier()

                    #==== Bond orders ====#
                    if bo_pairs is not None:
                        bo_pairs_ = tuple( [ (bo_pairs[i][0]-1,bo_pairs[i][1]-1) for i in
                                             range(0,len(bo_pairs)) ] )  # Transform to 0-base indices.
                        bo_mul, bo_low = \
                            bond_order.calc_pair(self.mol, dm_full, orbs, bo_pairs_,
                                                 self.ovl_ao)
                        if self.mpi is None or self.mpi.rank == 0:
                            bo_print.print_bo(tt, bo_low)
                        if self.mpi is not None: self.mpi.barrier()
                    
                    #==== Multipole components ====#
                    e_dpole, n_dpole, e_qpole, n_qpole = \
                        mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm_full, orbs)
                    if self.mpi is None or self.mpi.rank == 0:
                        mp_print.print_mpole(tt, e_dpole, n_dpole, e_qpole, n_qpole)
                    if self.mpi is not None: self.mpi.barrier()

                    issampled[i_sp] = True
                    i_sp += 1

                #==== Update logbook file ====#
                if self.mpi is None or self.mpi.rank == 0:
                    util_logbook.save(logbook, logbook['myname'], False, verbose=3)
                if self.mpi is not None: self.mpi.barrier()


        t_end_tevo = time.time()
        logbook.update({'t_end_tevo':t_end_tevo})
        _print('Time evolution takes ' + format_timespan(t_end_tevo-t_begin_tevo, max_units=5))

        
        #==== Print max min imaginary parts (for debugging) ====#
        if t_sample is not None:
            if self.mpi is None or self.mpi.rank == 0:
                q_print.footer()

        if t_sample is not None and bo_pairs is not None:
            if self.mpi is None or self.mpi.rank == 0:
                bo_print.footer()

        if t_sample is not None:
            if self.mpi is None or self.mpi.rank == 0:
                mp_print.footer()
                
        return logbook    
    ##############################################################
    

    ##############################################################
    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        b2.release_memory()

##############################################################




# 1) Does the input ket have to be complex or real?
# 2) Is bond_dims just a list of one element?
# 3) Why is hermitian False by default?




# A section from gfdmrg.py about the definition of the dmrg_mo_gf function has been
# removed.


# A section from gfdmrg.py has been removed.







#In ft_tddmrg.py, what does MYTDDMRG.fmt_size mean? This quantity exist inside the MYTDDMRG class
#definition. Shouldn't it be self.fmt_size.
#It looks like the RT_MYTDDMRG inherits from FTDMRG class, but is there no super().__init__ method in
#its definition?
#What does these various deallocate actually do? Somewhere an mps is deallocated (by calling mps.deallocate())
#but then the same mps is still used to do something.
#ANS: ME writes/reads mps to/from disk that's why mps can be deallocated although it is used again later.
