import numpy as np
from scipy.linalg import eigh
from TDDMRG_CM.observables.pcharge import get_atom_range


##########################################################
def calc(mol, pdm, mo, ovl=None):
    complex_pdm = (type(pdm[0,0,0]) == np.complex128)
    dtype = np.complex128 if complex_pdm else np.float64
    
    bo_mul = np.zeros((mol.natm,mol.natm), dtype=dtype)
    bo_low = np.zeros((mol.natm,mol.natm), dtype=dtype)
    for i in range(0, mol.natm):
        for j in range(i+1, mol.natm):
            a, b = calc_pair(mol, pdm, mo, ((i,j),), ovl)  # a, b are length-1 vectors because ((i,j),) is length-1 tuple.
            bo_mul[i,j], bo_low[i,j] = (a[0], b[0])
            bo_mul[j,i], bo_low[j,i] = bo_mul[i,j], bo_low[i,j]

    return bo_mul, bo_low
##########################################################


##########################################################
def calc_pair(mol, pdm, mo, atom_pairs, ovl=None):
    '''
    mol = Mole object.
    pdm = The complete (core+active) one-particle-reduced density matrix in MO rep.
    mo = The MOs in AO rep.
    atom_pairs = A sequence of 2-element sequence, e.g. ((0,2), (1,0), (3,1)), which contains
                 the indices of atom pairs between which the bond order is calculated.
    ovl = The overlap matrix associated with the AO basis defined in mol.
    '''    

    #==== Complex or real PDM? ====#
    assert len(pdm.shape) == 3, 'partial_charge.calc: pdm is not a 3D array.'
    assert len(mo.shape) == 3, 'partial_charge.calc: mo is not a 3D array.'
    assert pdm.shape[1] == pdm.shape[2], 'partial_charge.calc: pdm is not square.'
    complex_pdm = (type(pdm[0,0,0]) == np.complex128)
    dtype = np.complex128 if complex_pdm else np.float64
    
    #==== Setting up the system ====#
    nao = mol.nao
    atom_ao_range = get_atom_range(mol)
    
    #==== AO overlap matrix ====#
    if ovl is None:
        ovl = mol.intor('int1e_ovlp')
    es, U = eigh(ovl)
    ovl_half = U @ (np.diag( np.sqrt(es) ) @ U.conj().T)
        
    #==== P in AO basis ====#
    P = np.zeros((nao, nao), dtype=dtype)
    for i in range(0,2):
        P = P + mo[i,:,:] @ (pdm[i,:,:] @ mo[i,:,:].T)
        
    #==== Calculate bond orders ====#
    Bmul = np.einsum('ij, jk -> ik', P, ovl)
    bo_mul = np.zeros(len(atom_pairs), dtype=dtype)
    Blow = np.einsum('ij, jk, kl -> il', ovl_half, P, ovl_half)
    bo_low = np.zeros(len(atom_pairs), dtype=dtype)
    for i in range(0,len(atom_pairs)):
        at1, at2 = atom_pairs[i]

        #== AO ID range in atom A ==#
        ia_1 = atom_ao_range[at1][0]
        ia_2 = atom_ao_range[at1][1]

        #== AO ID range in atom B ==#
        ib_1 = atom_ao_range[at2][0]
        ib_2 = atom_ao_range[at2][1]

        bo_mul[i] = 0.0
        bo_low[i] = 0.0
        for j in range(ia_1, ia_2+1):
            for k in range(ib_1, ib_2+1):
                #==== Mulliken BO ====#
                bo_mul[i] = bo_mul[i] + Bmul[j,k] * Bmul[k,j]

                #==== Lowdin BO ====#
                bo_low[i] = bo_low[i] + np.abs(Blow[j,k])**2
    
    return bo_mul, bo_low
##########################################################
