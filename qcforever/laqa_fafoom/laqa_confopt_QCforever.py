# -*- coding: utf-8 -*-

import sys
import time
import datetime
import shutil

from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolDescriptors

from qcforever import laqa_fafoom


def LAQA_confopt_main(infilename, TotalCharge, SpinMulti, method, nproc, mem):

    PreInput = infilename.split('.')

    t_laqaopt_bgn = time.time()
    print(f"\nStart LAQA conformation search job at {datetime.datetime.now()}")
    if PreInput[-1] == "sdf":
        sdfmol = Chem.SDMolSupplier(infilename, removeHs=False)
        mols = [x for x in sdfmol if x is not None]
        mol = mols[0]
    elif PreInput[-1] == "xyz":
        mol = Chem.MolFromXYZFile(infilename)
        rdDetermineBonds.DetermineBonds(mol)
    else:
        exit()

    #get the number of rotatable bonds
    RotBond = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

    SMILES = Chem.MolToSmiles(mol)

    make_laqa_input(SMILES, SpinMulti, TotalCharge, RotBond, method, nproc, mem)

    laqa_fafoom.initgeom.LAQA_initgeom('laqa_setting.inp', SMILES)
    laqa_fafoom.laqa_optgeom.LAQA_optgeom('laqa_setting.inp')
	
    t_laqaopt_end = time.time()
    print(f"\nFinish LAQA conforation searh job at {datetime.datetime.now()}")
    print(f"Wall time of LAQA conformation optimization job: {t_laqaopt_end - t_laqaopt_bgn:20.2f} sec.")


def make_laqa_input(SMILES, SpinMulti, TotalCharge, RotBond, method, nproc, mem):

    # Smart population sizing with bounds to handle edge cases and large molecules
    # - Minimum of 10 to handle rigid molecules (RotBond=0)
    # - Tiered scaling to balance exploration vs computational cost
    # - Maximum cap to prevent excessive computation for very flexible molecules
    if RotBond <= 3:
        Num_popsize = 10  # Rigid/semi-rigid molecules: use default minimum
    elif RotBond <= 10:
        Num_popsize = min(15 + 2*RotBond, 35)  # Small-medium molecules: moderate scaling
    elif RotBond <= 20:
        Num_popsize = min(35 + RotBond, 55)  # Medium-large molecules: reduced scaling
    else:
        # Very large molecules: logarithmic scaling with cap at 70
        import math
        Num_popsize = min(55 + int(math.log2(RotBond - 19) * 5), 70)

    print(f"Number of rotatable bonds: {RotBond}")
    print(f"Initial population size: {Num_popsize}")

    input_s = ''

    input_s += '[Molecule]\n'
    input_s += f'\nsmiles = "{SMILES}"\n'

    input_s += f'\n[Initial geometry]\n'
    input_s += f'popsize = {Num_popsize}\n'

    # For large molecules, increase max iterations for structure generation
    if RotBond > 15:
        input_s += f'cnt_max = {min(1000, 500 + RotBond * 20)}\n'

    input_s += '\n[LAQA settings]\n'

    input_s += f'\ncharge = {TotalCharge}\n'
    input_s += f'mult = {SpinMulti}\n'


    if method == 'xtb':
        input_s += f'\nenergy_function = "{method}"\n'
        input_s += f'gfn = "2"\n'
    if method == 'pm6':
        input_s += '\nenergy_function = "g16"\n'
        #Get the directry that include 'g16'
        Gaussian_exedir = shutil.which('g16')
        #Delete the final binary file name of 'g16'
        input_s += f'gauss_exedir  = "{Gaussian_exedir[:-4]}"\n'
        input_s += f'qcmethod = "{method}" \n'
        if nproc > 1:
            input_s += f'nprocs = "{nproc}" \n'
        if mem != '':
            input_s += f'memory = "{mem}" \n'

    with open('laqa_setting.inp', 'w') as laqa_infile:
        
            laqa_infile.write(input_s)


