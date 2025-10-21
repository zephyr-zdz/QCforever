# -*- coding: utf-8 -*-

"""
Global geometry optimization code using
Look Ahead based on Quadratic Approximation (LAQA) method

Coded on Feb 10 2021

@Author: Michio Katouda (RIST) base on the LAQA code
coded by Kei Terayama (Univ. Tokyo, RIKEN AIP, and Kyoto Univ.)
e-mail: katouda@rist.or.jp
"""

import sys
import os.path
import subprocess
import configparser
import time
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from rdkit import Chem

from qcforever import laqa_fafoom


def LAQA_read_input(param_file):

    input_params_default = {'dir_inp': os.getcwd(), 'sdf_inp': 'initial_structures.sdf',
                            'sdf_out': 'optimized_structures.sdf',
                            'csv_out': 'optimized_structures.csv',
                            'struct_conv': 10,
                            'pool_begin': None, 'pool_end': None,
                            'macro_opt_cycle': 1000, 'micro_opt_cycle': 5,
                            'thr_econv': 5.0e-4, 'thr_fconv': 1.0e-6,
                            'printlevel': 0, 'energy_function': 'xtb',
                            'gauss_exedir': '/fefs/opt/x86_64/Gaussian/g16/',
                            'gauss_scrdir': os.getcwd(), 'g16_template': 'g16.template',
                            'qcmethod' : 'pm6',
                            'xtb_call': 'xtb', 'gfn': '2',
                            'charge': 0, 'mult': 1,
                            'solvmethod': None, 'solvent': 'water',
                            'nprocs': 1, 'memory': '3GB',
                            'parallel_init': True, 'n_parallel': None}

    input_params = laqa_fafoom.utilities.file2dict(param_file, ['LAQA settings'])
    input_params = laqa_fafoom.utilities.set_default(input_params, input_params_default)
    print('input_params:', input_params)

    input_params['energy_function'] = input_params['energy_function'].lower()
    input_params['inp_base'] = os.path.splitext(os.path.basename(input_params['sdf_inp']))[0]
    input_params['dir_out'] = input_params['dir_inp']
    input_params['out_base'] = os.path.splitext(os.path.basename(input_params['sdf_out']))[0]

    # Read input SDF file to get RDKit mol objects

    sdf_inp_file_path = input_params['dir_inp'] + '/' + input_params['sdf_inp']
    mols = [mol for mol in Chem.SDMolSupplier(sdf_inp_file_path, removeHs=False) if mol is not None]
    if input_params['pool_begin'] is None or input_params['pool_end'] is None:
        input_params['pool_begin'] = 1
        input_params['pool_end'] = len(mols)
    mols = mols[input_params['pool_begin']-1:input_params['pool_end']]
    print('len(mols): ', len(mols))

    print("\n<LAQA options>")
    print("struct_conv:", input_params['struct_conv'])
    print("pool_begin:", input_params['pool_begin'], "pool_end:", input_params['pool_end'])
    print("macro_opt_cycle:", input_params['macro_opt_cycle'])
    print("thr_econv:", input_params['thr_econv'], "thr_econv:", input_params['thr_fconv'])
    print("printlevel:", input_params['printlevel'])
    print("energy_function: ", input_params['energy_function'])
    if input_params['energy_function'] == "g16" or input_params['energy_function'] == "gaussian":
        print("gauss_exedir;", input_params['gauss_exedir'], "g16_template:", input_params['g16_template'])
    if input_params['energy_function'] == "xtb":
        print("xtb_call;", input_params['xtb_call'], "gfn:", input_params['gfn'])

    print("<Input file information>")
    print("Input file dir: ", input_params['dir_inp'], " Input SDF file: ", input_params['sdf_inp'])
    print("<Output file information>")
    print("Output file dir: ", input_params['dir_out'], " Output SDF file: ", input_params['sdf_out'])

    return input_params, mols


def calc_mae_rms_max_force(force):

    norms = np.array([np.linalg.norm(fo) for fo in force])

    mae_f = np.mean(norms)
    rms_f = np.sqrt(np.mean(norms**2))
    max_f = np.max(norms)

    return mae_f, rms_f, max_f


def evaluate_single_structure(args):
    """
    Evaluate a single structure for LAQA initialization.

    This function is designed to be called in parallel via multiprocessing.

    Args:
        args: tuple containing (j, mol, energy_function, params_dict)

    Returns:
        tuple: (j, energy, mae_f, score)
    """
    j, mol, energy_function, params = args

    try:
        print(f"Evaluating structure {j} in parallel...")

        sdf_string = Chem.MolToMolBlock(mol)

        # Perform gradient calculation
        if energy_function == "g16" or energy_function == "gaussian":
            energy, force, _ = laqa_fafoom.pyg16.g16_exec(
                sdf_string, params['gauss_exedir'], params['gauss_scrdir'],
                params['nprocs_per_struct'], params['memory'],
                jobtype='gradient',
                charge=params['charge'], mult=params['mult'],
                qcmethod=params['qcmethod'])
        elif energy_function == "xtb":
            energy, force = laqa_fafoom.pyxtb.xtb_exec(
                sdf_string, params['xtb_call'], 'gradient',
                params['gfn'], params['charge'], params['mult'],
                params['micro_opt_cycle'],
                params['solvmethod'], params['solvent'])
        else:
            raise ValueError(f"Unknown energy_function: {energy_function}")

        # Calculate force metrics
        mae_f, rms_f, max_f = calc_mae_rms_max_force(force)

        # Calculate LAQA score
        num_atoms = mol.GetNumAtoms()
        normfac_ene = 1.0 / float(num_atoms)
        dF = 1.0
        score = energy * normfac_ene - 1.0 * mae_f**2 / (2.0 * dF)

        print(f"Structure {j}: E={energy:.6f}, MAE_F={mae_f:.6e}, score={score:.6f}")

        return (j, energy, mae_f, score, True)

    except Exception as e:
        print(f"ERROR evaluating structure {j}: {e}")
        return (j, 0.0, 0.0, 0.0, False)


def LAQA_do_opt(input_params, mols):

    # Set input parameters

    inp_base = input_params['inp_base']
    sdf_inp = input_params['sdf_inp']
    struct_conv = input_params['struct_conv']
    macro_opt_cycle = input_params['macro_opt_cycle']
    micro_opt_cycle = optsteps = input_params['micro_opt_cycle']
    thr_econv = input_params['thr_econv']
    thr_fconv = input_params['thr_fconv']
    printlevel = input_params['printlevel']
    energy_function = input_params['energy_function']
    gauss_exedir = input_params['gauss_exedir']
    gauss_scrdir = input_params['gauss_scrdir']
    qcmethod = input_params['qcmethod']
    xtb_call = input_params['xtb_call']
    gfn = input_params['gfn']
    charge = input_params['charge']
    mult = input_params['mult']
    solvmethod = input_params['solvmethod']
    solvent = input_params['solvent']
    nprocs = input_params['nprocs']
    memory = input_params['memory']

    # Read input SDF file to get RDKit mol objects

    num_structures = len(mols)
    num_atoms = mols[0].GetNumAtoms()
    normfac_ene = 1.0 / float(num_atoms)
    print("\nStructure data\nNumber of structures: ", num_structures, "Number of atoms: ", num_atoms)

    opt_count = 0

    opt_struct_list = {}

    energy_list = []
    mae_f_list = []

    score_idx_calced = []
    score_calced = np.array( [] )

    score_idx_pool = []
    score_pool = np.array( [] )

    # Determine parallelization strategy
    parallel_init = input_params.get('parallel_init', True)
    n_parallel = input_params.get('n_parallel', None)

    # Validate and normalize parallel_init parameter
    if isinstance(parallel_init, str):
        parallel_init = parallel_init.lower() in ('true', '1', 'yes')
    parallel_init = bool(parallel_init)

    # Validate and normalize n_parallel parameter
    if n_parallel == "auto" or n_parallel == "Auto" or n_parallel == "AUTO":
        n_parallel = None  # Normalize to None for auto mode
    elif n_parallel is not None:
        try:
            n_parallel = int(n_parallel)
            if n_parallel < 1:
                raise ValueError(f"n_parallel must be positive, got {n_parallel}")
            if n_parallel > cpu_count():
                print(f"WARNING: n_parallel ({n_parallel}) exceeds available cores ({cpu_count()})")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid n_parallel value '{n_parallel}': must be None, 'auto', or a positive integer") from e

    # Calculate optimal parallelization
    total_cores = cpu_count()
    if n_parallel is None:
        # Auto-determine: balance between structure-level and QC-level parallelism
        if num_structures >= 4:
            # Many structures: prefer structure-level parallelism
            # Ensure we don't create more workers than structures
            n_parallel = min(num_structures, max(1, total_cores // 2))
            nprocs_per_struct = max(1, total_cores // n_parallel)
        else:
            # Few structures: use QC-level parallelism
            n_parallel = 1
            nprocs_per_struct = nprocs
    else:
        # Manual mode: user specified n_parallel
        if n_parallel > num_structures:
            print(f"WARNING: n_parallel ({n_parallel}) > number of structures ({num_structures})")
            print(f"         Adjusting n_parallel to {num_structures}")
            n_parallel = num_structures
        nprocs_per_struct = max(1, total_cores // n_parallel)

    print(f"\nParallelization strategy:")
    print(f"  Total CPU cores: {total_cores}")
    print(f"  Parallel structures: {n_parallel}")
    print(f"  Cores per structure: {nprocs_per_struct}")
    print(f"  Parallel initialization: {parallel_init}\n")

    # Initialization of LAQA optimization for energy target structure

    print("\nStart initialization of LAQA geometry optimization\n")

    if parallel_init and n_parallel > 1:
        # PARALLEL INITIALIZATION
        print(f"Using PARALLEL initialization with {n_parallel} workers\n")

        # Prepare parameters for worker processes
        worker_params = {
            'gauss_exedir': gauss_exedir,
            'gauss_scrdir': gauss_scrdir,
            'nprocs_per_struct': nprocs_per_struct,
            'memory': memory,
            'qcmethod': qcmethod,
            'xtb_call': xtb_call,
            'gfn': gfn,
            'charge': charge,
            'mult': mult,
            'micro_opt_cycle': micro_opt_cycle,
            'solvmethod': solvmethod,
            'solvent': solvent
        }

        # Create work items
        work_items = [(j, mols[j], energy_function, worker_params) for j in range(num_structures)]

        # Execute in parallel
        with Pool(processes=n_parallel) as pool_workers:
            results = pool_workers.map(evaluate_single_structure, work_items)

        # Process results
        for j, energy, mae_f, score, success in results:
            if success:
                energy_list.append(energy)
                mae_f_list.append(mae_f)
                score_pool = np.append(score_pool, score)
                score_idx_pool.append([j, 0])
                score_idx_calced.append([j, 0])
                score_calced = np.append(score_calced, score)
            else:
                print(f"WARNING: Structure {j} failed evaluation, skipping")

    else:
        # SEQUENTIAL INITIALIZATION (original code)
        print("Using SEQUENTIAL initialization\n")

        for j in range(num_structures):

            print("LAQA initialization: ID {:>6}".format(j))

            # Perform simulation program (quantum chemistry or force field)

            sdf_string = Chem.MolToMolBlock(mols[j])
            if energy_function == "g16" or energy_function == "gaussian":
                energy, force, sdf_string_opt = laqa_fafoom.pyg16.g16_exec(sdf_string, gauss_exedir, gauss_scrdir,
                                                         nprocs, memory, jobtype='gradient',
                                                         charge=charge, mult=mult,
                                                         qcmethod=qcmethod)
            elif energy_function == "xtb":
                jobtype='gradient'
                energy, force = laqa_fafoom.pyxtb.xtb_exec(sdf_string, xtb_call, jobtype,
                                         gfn, charge, mult, optsteps, solvmethod, solvent)
            else:
                print("NYI for code: ", energy_function)
                sys.exit(1)

            # Obtaion total energy E and MAE force F

            mae_f, rms_f, max_f = calc_mae_rms_max_force(force)
            print("Energy: {:>15.8}".format(energy))
            print("MAE force: {:>15.8e} RMS force: {:>15.8e} Max force: {:>15.8}"\
                  .format(mae_f, rms_f, max_f))
            energy_list.append(energy)
            mae_f_list.append(mae_f)

            # Calculate scoreing function for LAQA optimization

            dF = 1.0
            score = energy * normfac_ene - 1.0 * mae_f**2 / (2.0 * dF)
            print('E:', energy, 'F:', mae_f, 'dF:', dF, 'score:', score)

            score_pool = np.append(score_pool, score)
            score_idx_pool.append([j, 0])
            score_idx_calced.append([j, 0])
            score_calced = np.append(score_calced, score_pool[j])

    print("\nEnd initialization of LAQA geometry optimization\n\n")

    print("Summary of initialization of LAQA optimization\n"\
        + "[ image]           score           energy[au]        MAE force[au]")
    for j in range(len(energy_list)):
        print("[{:>6}] {:>15.8f} {:>20.10f} {:>20.10f}"\
            .format(j, score_pool[j], energy_list[j], mae_f_list[j]))

    # Perform LAQA optimization (Main loop)

    for cycle in range(macro_opt_cycle):

        print("\nLAQA geometry optimization: {:>6} cycle ".format(cycle))

        # Selection of structure to be performed local geometry optimization
        # ID with minimum value of LAQA score function is selected.

        arm = np.argmin(score_pool)
        score_pool = np.delete(score_pool, arm)
        j, i = score_idx_pool.pop(arm)
        i += 1
        # Perform simulation program (quantum chemistry or force field)

        sdf_string = Chem.MolToMolBlock(mols[j])
        if energy_function == "g16" or energy_function == "gaussian":
            energy, force, sdf_string_opt = laqa_fafoom.pyg16.g16_exec(sdf_string, gauss_exedir, gauss_scrdir,
                                                     nprocs, memory, jobtype='opt',
                                                     charge=charge, mult=mult,
                                                     qcmethod=qcmethod,
                                                     optsteps=micro_opt_cycle)
        elif energy_function == "xtb":
            jobtype='opt'
            energy, xyz_string_opt = laqa_fafoom.pyxtb.xtb_exec(sdf_string, xtb_call, jobtype,
                            gfn, charge, mult, optsteps, solvmethod, solvent)
            jobtype='gradient'
            sdf_string_opt = laqa_fafoom.utilities.xyz2sdf(xyz_string_opt, sdf_string)
            energy, force = laqa_fafoom.pyxtb.xtb_exec(sdf_string_opt, xtb_call, jobtype,
                                     gfn, charge, mult, optsteps, solvmethod, solvent)
        else:
            print("NYI for computational chemistry code: ", energy_function)
            sys.exit(1)
        mols[j] = Chem.MolFromMolBlock(sdf_string_opt, removeHs=False)

        # Obtaion total energy E and MAE force F

        mae_f, rms_f, max_f = calc_mae_rms_max_force(force)
        print("Energy: {:>15.8}".format(energy))
        print("MAE force: {:>15.8e} RMS force: {:>15.8e} Max force: {:>15.8}"\
              .format(mae_f, rms_f, max_f))

        # Calculate scoreing function for LAQA optimization

        energy_m1 = energy_list[j]
        mae_f_m1 = mae_f_list[j]

        dF = abs(mae_f - mae_f_m1)
        dF = 1e-6 if dF == mae_f else dF
        #score = energy * normfac_ene - 1.0 * mae_f**2 / 2.0
        score = energy * normfac_ene - 1.0 * mae_f**2 / (2.0 * dF)
        print('E:', energy, 'F:', mae_f, 'dF:', dF, 'score:', score)

        energy_list[j] = energy
        mae_f_list[j] = mae_f
        score_idx_calced.append([j, i])
        score_calced = np.append(score_calced, score)

        print("Cycle {:>6} [{:>6},{:>6}] {:>15.8f} {:>20.10f} {:>20.10f}"\
            .format(cycle, j, i, score, energy, mae_f))

        # LAQA global optimization congergence investigation

        if abs(energy - energy_m1) > thr_econv:
            score_pool = np.append(score_pool, score)
            score_idx_pool.append([j, i])
        else:
            print("LAQA local geometry optpmization is converged: structure ID {:>6}".format(j+1))
            opt_count += 1
            opt_struct_list[j+1] = energy
            
            if opt_count == struct_conv or len(score_pool) == 0:
                last_cycle = cycle
                break
    
    # Post processing of LAQA optimization

    if opt_count == struct_conv or len(score_pool) == 0:
        print("\nLAQA global geometry optimization is converged at step {:>6}"\
              .format(last_cycle))
    else:
        print("\nLAQA global geometry optimization is not converged at step {:>6}"\
              .format(cycle))
    print("Total number of LAQA optimized structrure found: {:>6}".format(opt_count))

    return mols, opt_struct_list


def LAQA_print_summary(input_params, mols, opt_struct_list):

    # Physical constant
    hartree2eV = 27.21138602

    print("\nOptimized structure list (unsorted)")
    print("{:>10} {:>20} {:>20}".format("Struct. ID", "Total energy [au]", "Total energy [eV]"))
    for id, energy in opt_struct_list.items():
        print("{:>10} {:>20.8f} {:>20.8f}".format(id, energy, energy*hartree2eV))

    print("\nOptimized structure list (sorted by lower energy)")
    print("{:>10} {:>20} {:>20}".format("Struct. ID", "Total energy [au]", "Total energy [eV]"))
    for id, energy in sorted(opt_struct_list.items(), key=lambda x: x[1]):
        print("{:>10} {:>20.8f} {:>20.8f}".format(id, energy, energy*hartree2eV))

    out_csv_path = input_params['dir_out'] + '/' + input_params['csv_out']
    with open(out_csv_path, "w") as f:
        f.write('\"Structure ID","Total energy [au]"\n')
        for id, energy in sorted(opt_struct_list.items(), key=lambda x: x[1]):
            f.write("{},{}\n".format(id, energy))

    # Write optimized strucures to a SDF file
    sdf_out_path = input_params['dir_out'] + '/' + input_params['sdf_out']
    sdf_out = Chem.SDWriter(sdf_out_path)
    for id, energy in sorted(opt_struct_list.items(), key=lambda x: x[1]):
        sdf_out.write(mols[id-1])


def LAQA_optgeom(param_file):

    t_laqa_bgn = time.time()
    print("Start LAQA geometry optimization job at ",
          datetime.datetime.now(), '\n')

    # Read LAQA parameter and geometory input files

    input_params, mols = LAQA_read_input(param_file)

    # Perform LAQA geometory optpmization

    mols, opt_struct_list = LAQA_do_opt(input_params, mols)

    # Write results of LAQA geometory optpmization

    if len(opt_struct_list) > 0:
        LAQA_print_summary(input_params, mols, opt_struct_list)

    t_laqa_end = time.time()
    print("\nFinish LAQA geometry optimization job at ",
          datetime.datetime.now())
    print("Wall time of LAQA geometry optimization job:     {:20.2f} sec."\
          .format(t_laqa_end - t_laqa_bgn))


if __name__ == "__main__":

    param_file = sys.argv[1]
    LAQA_optgeom(param_file)
