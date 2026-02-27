#import orbs_generate.defvals as defvals
import defvals


def get_inputs(inp_file):

    exec(open(inp_file).read(), globals())
    inputs = {}

    #==== General parameters ====#
    inputs['atoms'] = atoms
    inputs['basis'] = basis
    inputs['charge'] = charge
    inputs['twosz'] = twosz

    try:
        inputs['init_orbs'] = init_orbs
    except NameError:
        inputs['init_orbs'] = defvals.def_init_orbs
    try:
        inputs['init_occs'] = init_occs
    except NameError:
        inputs['init_occs'] = defvals.def_init_occs
    try:
        inputs['init_ergs'] = init_ergs
    except NameError:
        inputs['init_ergs'] = defvals.def_init_ergs
    try:
        inputs['init_rdm'] = init_rdm
    except NameError:
        inputs['init_rdm'] = defvals.def_init_rdm
    try:
        inputs['dump_inputs'] = dump_inputs
    except NameError:
        inputs['dump_inputs'] = defvals.def_dump_inputs
    try:
        inputs['prefix'] = prefix
    except NameError:
        if inp_file[len(inp_file)-3:len(inp_file)] == '.py':
            inputs['prefix'] = inp_file[0:len(inp_file)-3]
        else:
            inputs['prefix'] = inp_file
    try:
        inputs['inp_ecp'] = inp_ecp
    except NameError:
        inputs['inp_ecp'] = defvals.def_inp_ecp
    try:
        inputs['group'] = group
    except NameError:
        inputs['group'] = defvals.def_group
    try:
        inputs['save_dir'] = save_dir
    except NameError:
        inputs['save_dir'] = defvals.def_save_dir
    try:
        inputs['natorb'] = natorb
    except NameError:
        inputs['natorb'] = defvals.def_natorb
    try:
        inputs['conv_tol'] = conv_tol
    except NameError:
        inputs['conv_tol'] = defvals.def_conv_tol
    try:
        inputs['canonicalize'] = canonicalize
    except NameError:
        inputs['canonicalize'] = defvals.def_canonicalize

        
    #==== CAS parameters ====#
    try:
        inputs['source'] = source
    except NameError:
        inputs['source'] = defvals.def_source
        
    if inputs['source'] == 'rhf':
        # All inputs used by RHF are also used by CASSCF.
        pass
    elif inputs['source'] == 'casscf':
        inputs['nCAS'] = nCAS
        inputs['nelCAS'] = nelCAS

        try:
            inputs['frozen'] = frozen
        except NameError:
            inputs['frozen'] = defvals.def_frozen
        try:            
            inputs['ss'] = ss
        except NameError:
            inputs['ss'] = defvals.def_ss
        try:            
            inputs['ss_shift'] = ss_shift
        except NameError:
            inputs['ss_shift'] = defvals.def_ss_shift
        try:
            inputs['wfnsym'] = wfnsym
        except NameError:
            inputs['wfnsym'] = defvals.def_wfnsym
        try:
            inputs['fcisolver'] = fcisolver
        except NameError:
            inputs['fcisolver'] = defvals.def_fcisolver
        try:
            inputs['init_basis'] = init_basis
        except NameError:
            inputs['init_basis'] = defvals.def_init_basis
        try:
            inputs['set_cas_sym'] = set_cas_sym
        except NameError:
            inputs['set_cas_sym'] = defvals.def_set_cas_sym
        try:
            inputs['state_average'] = state_average
        except NameError:
            inputs['state_average'] = defvals.def_state_average
        try:
            inputs['sa_weights'] = sa_weights
        except NameError:
            inputs['sa_weights'] = defvals.def_sa_weights            
        try:
            inputs['sort_out'] = sort_out
        except NameError:
            inputs['sort_out'] = defvals.def_sort_out

        #==== DMRGSCF ====#
        try:
            inputs['max_bond_dim'] = max_bond_dim
        except NameError:
            inputs['max_bond_dim'] = defvals.def_max_bond_dim
        try:
            inputs['sweep_tol'] = sweep_tol
        except NameError:
            inputs['sweep_tol'] = defvals.def_sweep_tol
        try:
            inputs['b2_extra'] = b2_extra
        except NameError:
            inputs['b2_extra'] = defvals.def_b2_extra

    elif inputs['source'] == 'dft':
        inputs['xc'] = xc

    try:
        inputs['sorting'] = sorting
        if inputs['sorting']['type'] == 'reference':
            inputs['sorting']['orb_ref'] = sorting['orb_ref']

        # sorting['range']: A tuple of two 1-based integers.
        try:
            inputs['sorting']['range'] = sorting['range']
        except KeyError:
            inputs['sorting']['range'] = defvals.def_sorting['range']
        try:
            inputs['sorting']['similar_thr'] = sorting['similar_thr']
        except KeyError:
            inputs['sorting']['similar_thr'] = defvals.def_sorting['similar_thr']
        try:
            inputs['sorting']['dissimilar_break'] = sorting['dissimilar_break']
        except KeyError:
            inputs['sorting']['dissimilar_break'] = defvals.def_sorting['dissimilar_break']
        try:
            inputs['sorting']['irrep_sort'] = sorting['irrep_sort']
        except KeyError:
            inputs['sorting']['irrep_sort'] = defvals.def_sorting['irrep_sort']
    except NameError:
        inputs['sorting'] = defvals.def_sorting
            
    try:
        inputs['localize'] = localize
    except NameError:
        inputs['localize'] = defvals.def_localize
        
    if inputs['localize']:
        # loc_subs:
        #   A list of lists where each list element contains the 1-base orbital indices
        #   to be localized in the localization subspace represented by this list element.
        #   That is, the lists in loc_subs represent the localization subspaces.
        try:
            inputs['loc_subs'] = loc_subs
        except NameError:
            inputs['loc_subs'] = defvals.def_loc_subs
            
        # loc_occs:
        #   The occupation thresholds that define the localization subspaces.
        #   If there are more than two subspaces, it must be a list with
        #   monotonically increasing elements ranging between but does not
        #   include 0.0 and 2.0.
        try:
            inputs['loc_occs'] = loc_occs
        except NameError:
            inputs['loc_occs'] = defvals.def_loc_occs

        try:
            inputs['loc_type'] = loc_type
        except NameError:
            inputs['loc_type'] = 'DEFINE_LATER' #OLD ['PM'] * len(inputs['loc_subs'])
        try:
            inputs['loc_irrep'] = loc_irrep
        except NameError:
            inputs['loc_irrep'] = 'DEFINE_LATER' #OLD [True] * len(inputs['loc_subs'])
        try:
            inputs['loc_exclude'] = loc_exclude
        except NameError:
            inputs['loc_exclude'] = defvals.def_loc_exclude
        try:
            inputs['loc_sort'] = loc_sort
        except NameError:
            inputs['loc_sort'] = defvals.def_loc_sort
            
    return inputs
