#    Copyright 2015 Adriana Supady
#
#    This file is part of fafoom.
#
#   Fafoom is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   Fafoom is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#   along with fafoom.  If not, see <http://www.gnu.org/licenses/>.
''' Communicate between the structure and the degrees of freedom.'''
from __future__ import division
from rdkit import Chem
from rdkit.Chem import AllChem

from qcforever import laqa_fafoom


def get_atoms_and_bonds(smiles):
    """Build the molecule from SMILES and return the number of atoms and bonds.

    Args(required):
        smiles (str): one-line representation of the molecule
    Returns:
        Number of atoms, number of bonds
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("The smiles is invalid")
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms(), mol.GetNumBonds()


def get_positions(type_of_deg, smiles, **kwargs):
    """Find the positions (tuples of atom indicies) of the degrees of freedom.

    Args(required):
        type_of_deg (str)
        smiles (str)
        if cistrans should be optimized:
            smarts_cistrans
    Args(optimal):
        list_of_torsion (list)
        smarts_torsion (str)
        filter_smarts_torsion (str)
        list_of_cistrans (list)
        list_of_pyranosering (list)
    Returns:
        list of touples defining the positions of the degree of freedom
    """

    if type_of_deg == "torsion":
        if 'list_of_torsion' in kwargs:
            return laqa_fafoom.deg_of_freedom.Torsion.find(smiles, positions=kwargs['list_of_torsion'])
        else:
            if 'smarts_torsion' in kwargs:
                if 'filter_smarts_torsion' in kwargs:
                    return laqa_fafoom.deg_of_freedom.Torsion.find(smiles,
                                        smarts_torsion=kwargs['smarts_torsion'],
                                        filter_smarts_torsion=
                                        kwargs['filter_smarts_torsion'])
                else:
                    return laqa_fafoom.deg_of_freedom.Torsion.find(smiles,
                                        smarts_torsion=kwargs['smarts_torsion'])
            else:
                return laqa_fafoom.deg_of_freedom.Torsion.find(smiles)
    if type_of_deg == "cistrans":
        if 'list_of_cistrans' in kwargs:
            return laqa_fafoom.deg_of_freedom.CisTrans.find(smiles, positions=kwargs['list_of_cistrans'])
        else:
            return laqa_fafoom.deg_of_freedom.CisTrans.find(smiles,
                                 smarts_cistrans=kwargs['smarts_cistrans'])

    if type_of_deg == "pyranosering":
        if 'list_of_pyranosering' in kwargs:
            return laqa_fafoom.deg_of_freedom.PyranoseRing.find(smiles,
                                     positions=kwargs['list_of_pyranosering'])
        else:
            return laqa_fafoom.deg_of_freedom.PyranoseRing.find(smiles)


def create_dof_object(type_of_deg, positions):
    """Initialize the degree of freedom from the positions

    Args:
        type_of_deg (str)
        positsion (list)
    Returns:
        degree of freedom object
    """
    if type_of_deg == "torsion":
        return laqa_fafoom.deg_of_freedom.Torsion(positions)
    if type_of_deg == "cistrans":
        return laqa_fafoom.deg_of_freedom.CisTrans(positions)
    if type_of_deg == "pyranosering":
        return laqa_fafoom.deg_of_freedom.PyranoseRing(positions)


def generate_conformers_batch(smiles, num_conformers, distance_cutoff_1, distance_cutoff_2):
    """Generate multiple conformers efficiently using batch processing.

    This function uses RDKit's EmbedMultipleConfs for much faster generation
    compared to sequential embedding. Particularly beneficial for large molecules.

    Args:
        smiles (str): SMILES representation of the molecule
        num_conformers (int): Number of conformers to generate
        distance_cutoff_1 (float): min distance between non-bonded atoms [A]
        distance_cutoff_2 (float): max distance between bonded atoms [A]

    Returns:
        list of sdf strings: Valid conformer structures
    """
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Configure embedding parameters based on molecule size
    params = AllChem.ETKDGv3()
    params.randomSeed = -1  # Use different random seeds
    params.numThreads = 0  # Use all available cores
    params.pruneRmsThresh = 0.5 if num_rot_bonds > 5 else 0.3  # Auto-prune similar structures

    # Generate more conformers than needed to account for failures
    target_confs = int(num_conformers * 1.5)

    print(f"Generating {target_confs} conformers (target: {num_conformers})...")

    try:
        # Batch generation - much faster than sequential
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=target_confs,
            params=params
        )

        if len(conf_ids) == 0:
            print("Warning: Batch embedding failed, falling back to sequential method")
            return []

        print(f"Successfully embedded {len(conf_ids)} conformers")

        # Batch optimization using UFF
        print("Optimizing conformers...")
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=0)

        # Validate and collect valid conformers
        valid_conformers = []
        for i, conf_id in enumerate(conf_ids):
            if len(valid_conformers) >= num_conformers:
                break

            try:
                sdf_string = Chem.MolToMolBlock(mol, confId=conf_id)
                if laqa_fafoom.utilities.check_geo_sdf(sdf_string, distance_cutoff_1, distance_cutoff_2):
                    valid_conformers.append(sdf_string)
            except Exception as e:
                print(f"Warning: Conformer {i} validation failed: {e}")
                continue

        print(f"Generated {len(valid_conformers)} valid conformers")
        return valid_conformers

    except Exception as e:
        print(f"Error in batch conformer generation: {e}")
        return []


def template_sdf(smiles, distance_cutoff_1, distance_cutoff_2):
    """Create a template sdf string and writes it to file.

    Uses ETKDGv3 (Experimental Torsion-angle Knowledge Distance Geometry)
    for improved performance and chemical accuracy, especially for large molecules.

    Args(required):
        smiles (str): one-line representation of the molecule
    Args(optional):
        distance_cutoff_1 (float): min distance between non-bonded atoms [A]
        distance_cutoff_2 (float): max distance between bonded atoms [A]
    Returns:
        sdf string
    """
    from rdkit.Chem import rdMolDescriptors

    cnt = 0
    sdf_check = True
    max_attempts = 100  # Prevent infinite loops

    # Prepare molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Determine molecule size for method selection
    num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    use_etkdg = num_rot_bonds > 3  # Use advanced method for flexible molecules

    while sdf_check and cnt < max_attempts:
        try:
            if use_etkdg:
                # Use ETKDGv3 for better performance on larger molecules
                params = AllChem.ETKDGv3()
                params.randomSeed = -1  # Different random seed each time
                params.useRandomCoords = False  # Use knowledge-based coords
                embed_result = AllChem.EmbedMolecule(mol, params)
            else:
                # Use simpler method for small/rigid molecules
                embed_result = AllChem.EmbedMolecule(mol, useRandomCoords=True)

            if embed_result == -1:
                # Embedding failed, try again
                cnt += 1
                continue

            # Optimize geometry
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)

            # Generate SDF string and validate
            sdf_string = Chem.MolToMolBlock(mol)
            check = laqa_fafoom.utilities.check_geo_sdf(sdf_string, distance_cutoff_1, distance_cutoff_2)

            if check:
                sdf_check = False
                Chem.SDWriter('mol.sdf').write(mol)
            else:
                cnt += 1

        except Exception as e:
            print(f"Warning: Embedding attempt {cnt} failed: {e}")
            cnt += 1

    if cnt >= max_attempts:
        print(f"Warning: Reached maximum embedding attempts ({max_attempts})")
        print("Using last generated structure even if validation failed")

    return sdf_string
