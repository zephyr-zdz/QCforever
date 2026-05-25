import sys
import os
import time
import datetime

from qcforever import laqa_fafoom

import random
import numpy as np
from rdkit import Chem


def _write_population_sdf(population, sdf_out):
    """Write the selected initial population as valid SDF records."""
    writer = Chem.SDWriter(sdf_out)
    try:
        for struct in population:
            mol = Chem.MolFromMolBlock(struct.sdf_string, removeHs=False)
            if mol is None:
                raise ValueError("Could not parse optimized structure as SDF")
            if hasattr(struct, 'energy'):
                mol.SetProp("Energy", str(struct.energy))
            writer.write(mol)
    finally:
        writer.close()


def _optimize_candidate(str3d, energy_function, params, blacklist, population):
    if not str3d.is_geometry_valid():
        print("Geometry of " + str(str3d) + " is invalid.")
        return False

    if str3d in blacklist:
        print("Geomerty of " + str(str3d) + " is fine, but already known.")
        return False

    name = "initial_%d" % (len(population))
    try:
        laqa_fafoom.run_utilities.optimize(str3d, energy_function, params, name)
    except Exception as exc:
        print("Optimization of {} failed: {}".format(str(str3d), exc))
        return False

    laqa_fafoom.run_utilities.check_for_kill()
    str3d.send_to_blacklist(blacklist)
    population.append(str3d)
    print("Geometry of {} is added to the population, energy: {:.5f}"
          .format(str(str3d), str3d.energy))
    laqa_fafoom.run_utilities.relax_info(str3d)
    return True


def _structure_from_sdf(mol, sdf_string):
    str3d = laqa_fafoom.structure.Structure(mol)
    str3d.sdf_string = sdf_string
    for dof in str3d.dof:
        dof.update_values(str3d.sdf_string)
    return str3d


def _populate_batch(mol, params, energy_function, blacklist, population):
    popsize = params['popsize']
    overgenerate_factor = max(1.0, float(params['overgenerate_factor']))
    num_candidates = max(popsize, int(popsize * overgenerate_factor))

    conformers = laqa_fafoom.get_parameters.generate_conformers_batch(
        mol.smiles,
        num_candidates,
        mol.distance_cutoff_1,
        mol.distance_cutoff_2,
        overgenerate_factor=1.0,
        rmsd_prune=params['rmsd_prune'],
        seed=params['seed'])

    if not conformers:
        print("Batch conformer generation did not produce valid candidates.")
        return 0

    print("Optimizing {} batch-generated conformer candidates.".format(len(conformers)))
    accepted_before = len(population)
    for sdf_string in conformers:
        str3d = _structure_from_sdf(mol, sdf_string)
        _optimize_candidate(str3d, energy_function, params, blacklist, population)

    population[:] = sorted(population, key=lambda x: x.energy)[:popsize]
    return len(population) - accepted_before


def _populate_legacy(mol, params, energy_function, blacklist, population):
    popsize = params['popsize']
    cnt_max = params['cnt_max']
    cnt = 0

    while len(population) < popsize and cnt < cnt_max:
        print("\n----- New trial: {:6} -----\n".format(cnt+1))
        str3d = laqa_fafoom.structure.Structure(mol)
        str3d.generate_structure()
        _optimize_candidate(str3d, energy_function, params, blacklist, population)
        cnt += 1

    if cnt == cnt_max and len(population) < popsize:
        print("\nThe allowed number of trials for building the "
              "population has been exceeded.")


def LAQA_initgeom(param_file, SMILES=""):

    t_init_bgn = time.time()
    print(f"Start 3D structure initialization job at {datetime.datetime.now()}")

    # Decide for restart or a simple run: NYI
    #opt = simple_or_restart()

    # Build a dictionary from Initial Geometry settings of the parameter file.

    params = laqa_fafoom.utilities.file2dict(param_file, ['Initial geometry'])

    dict_default = {'popsize': 10, 'max_iter': 30,
                    'iter_limit_conv': 20, 'energy_diff_conv': 0.001,
                    'cnt_max': 500, 'seed': None,
                    'energy_function': 'ff',
                    'force_field': 'uff', 'optsteps': 1000,
                    'energy_tol': 1.0e-6, 'force_tol': 1.0e-4,
                    'xtb_call': 'xtb', 'gfn': '2',
                    'gauss_exedir': '~/bin/g16', 'gauss_scrdir': os.getcwd(),
                    'qcmethod': 'pm6', 'solvmethod': None, 'solvent': 'water',
                    'charge': 0, 'mult': 1,
                    'nprocs': 1, 'memory': '2GB',
                    'sdf_out': 'initial_structures.sdf', 'printlevel': 0,
                    'initgeom_method': 'batch_etkdg',
                    'overgenerate_factor': 2.0, 'rmsd_prune': None}

    # Set defaults for parameters not defined in the parameter file.

    params = laqa_fafoom.utilities.set_default(params, dict_default)
    print("params: ", params)
    energy_function = laqa_fafoom.run_utilities.detect_energy_function(params)
    popsize = params['popsize']
    cnt_max = params['cnt_max']
    seed = params['seed']
    printlevel = params['printlevel']
    initgeom_method = str(params['initgeom_method']).lower()
    print("energy_function: " , energy_function)
    print("popsize:", popsize, "cnt_max: ", cnt_max, "seed ", seed)
    print("initgeom_method:", initgeom_method,
          "overgenerate_factor:", params['overgenerate_factor'],
          "rmsd_prune:", params['rmsd_prune'])

    random.seed(seed)
    np.random.seed(seed=seed)

    # Set up template molecule.

    mol = laqa_fafoom.structure.MoleculeDescription(param_file)

################To get a SMILES string from an augment#########
    if SMILES != "":
        mol.smiles = SMILES
###############################################################

    print("SMILES: ", mol.smiles)
    print("distance_cutoff_1: ", mol.distance_cutoff_1)
    print("distance_cutoff_2: ", mol.distance_cutoff_2)

    # Assign the permanent attributes to the molecule.

    mol.get_parameters()
    mol.create_template_sdf()
    if printlevel > 0:
        print(mol.template_sdf_string)

    # Check for potential degree of freedom related parameters.

    linked_params = laqa_fafoom.run_utilities.find_linked_params(mol, params)
    print("Number of atoms: ", mol.atoms)
    print("Number of bonds: ", mol.bonds)

    for dof in mol.dof_names:
        print("Number of identified " + str(dof) + ": " +
              str(len(getattr(mol, dof))))
        print("Identified  " + str(dof) + ": " + str(getattr(mol, dof)))

    print("\n___Initialization of 3D structures ___\n")
    population, blacklist = [], []
    laqa_fafoom.utilities.remover_file(params['sdf_out'])

    if initgeom_method == 'batch_etkdg':
        _populate_batch(mol, params, energy_function, blacklist, population)
        if len(population) < popsize:
            print("Batch initialization produced {}/{} structures; falling back to legacy generation."
                  .format(len(population), popsize))
            _populate_legacy(mol, params, energy_function, blacklist, population)
    elif initgeom_method == 'legacy':
        _populate_legacy(mol, params, energy_function, blacklist, population)
    else:
        raise ValueError("Unknown initgeom_method: {}".format(params['initgeom_method']))

    if len(population) == 0:
        raise RuntimeError("Initial geometry generation failed: no valid structures were produced")

    population_sorted = sorted(population, key=lambda x:x.energy)[:popsize]
    _write_population_sdf(population_sorted, params['sdf_out'])

    print("\n___Initialization of 3D structure completed___\n")
    print("Total number of population (vaid structures): ", len(population_sorted))

    print("\nInitial population before sorting:\nID    Energy [eV]")
    for i in range(len(population)):
        print("{:5} {:20.6f}".format(population[i].index, population[i].energy))

    print("\nInitial population after sorting:\nID    Energy [eV]")
          
    for i in range(len(population_sorted)):
        print("{:5} {:20.6f}".format(population_sorted[i].index, population_sorted[i].energy))
    #print("\nBlacklist: " + ', '.join([str(v) for v in blacklist]))

    t_init_end = time.time()
    print("\nFinish 3D structure initialization job at ",\
          datetime.datetime.now())
    print("Wall time of initialization job:                 {:20.2f} sec."\
          .format(t_init_end - t_init_bgn))


if __name__ == '__main__':

    param_file = sys.argv[1]
    LAQA_initgeom(param_file)
