#Wrapper for xTB

import glob
import os
import shutil
import subprocess
import tempfile

from qcforever import laqa_fafoom

hartree2eV = 27.21138602
hartree2kcalmol = 627.503


class xTBObject():
    '''Create and handle xTB objects.'''
    def __init__(self, sdf_string, xtb_call, jobtype='opt', 
                 gfn='2', charge=0, mult=1, optsteps=200,
                 solvmethod=None, solvent='water',
                 sdf_out='optimized_structures.sdf', workdir=None):
                 
        """Initialize the xTBObject.
        Args(required):
            sdf_string (str): sdf file string
            xtb_call   (str): e.g. xTB for for parallel version
                                    /the/complete/path/to/xtb
        Args(optional):
            jobtype    (default='opt')
            gfn        (default=2)
            charge     (default=0)
            mult       (default=1)
            optsteps   (default=500)
            solvmethod (default=None)
            solvent    (default='water')
            sdf_out    (default=optimized_structures.sdf)
        Raises:
            KeyError: if the commandline or memory is not defined
        """
        self.sdf_string = sdf_string
        self.xtb_call = xtb_call
        self.jobtype = jobtype
        self.gfn = gfn
        self.charge = charge
        self.mult = mult
        self.optsteps = optsteps
        self.solvmethod = solvmethod
        self.solvent = solvent
        self.sdf_out = sdf_out
        self.workdir = workdir
        self._owns_workdir = workdir is None

    def _ensure_workdir(self):
        if self.workdir is None:
            self.workdir = tempfile.mkdtemp(prefix='qcforever_xtb_')
        else:
            os.makedirs(self.workdir, exist_ok=True)

    def _path(self, filename):
        self._ensure_workdir()
        return os.path.join(self.workdir, filename)

    def generate_input(self):
        """Create input files for xTB."""
#        with open('xtbin.mol', 'w') as f:
#            f.write(self.sdf_string)

        xyz_string = laqa_fafoom.utilities.sdf2xyz(self.sdf_string)
        with open(self._path('xtbin.xyz'), 'w') as f:
            f.write(xyz_string)

    def run_xtb(self):
        """Run xTB and write output to 'result.out'. The optimized
        geometry is written to 'xtbopt.mol'.

        Warning: this function uses subprocessing to invoke the run.
        The subprocess's shell is set to TRUE.
        Raises:
            OSError: if xtbin.mol not present in the working directory
        """
        #for defining OMP_STACKSIZE
        os.environ["OMP_STACKSIZE"] = "5G"

        success = False
        self._ensure_workdir()
        if os.path.exists(self._path('xtbin.xyz')) is False:
            raise OSError('Required input file not present.')

        if self.jobtype == 'gradient':
            com_xtb = self.xtb_call \
                      + ' --gfn{:>2} xtbin.xyz --chrg {} --uhf {} --grad'\
                          .format(self.gfn, self.charge, self.mult)
            if self.solvmethod is not None :
                com_xtb += ' --alpb {}'.format(self.solvent)
            with open(self._path('result.out'), 'w') as out:
                xtb = subprocess.run(com_xtb, stdout=out, stderr=subprocess.STDOUT,
                                     shell=True, cwd=self.workdir)
            if xtb.returncode != 0:
                raise RuntimeError("xTB gradient calculation failed with return code {}".format(xtb.returncode))
            if not os.path.exists(self._path('gradient')):
                raise RuntimeError("xTB gradient calculation did not produce a gradient file")

            with open(self._path('gradient'), 'r') as f:
                searchfile = f.readlines()
                num_atoms = int((len(searchfile) - 3) / 2)
                self.energy = float(searchfile[1].split()[6])
                grad = []
                for i in range(2+num_atoms, 2+2*num_atoms):
                    a = searchfile[i].split()
                    grad.append([float(a[0]), float(a[1]), float(a[2])])
                self.gradient = grad
                success = True

        else:
            com_xtb = self.xtb_call \
                      + ' --gfn{:>2} xtbin.xyz --chrg {} --uhf {} --opt --cycles {}'\
                          .format(self.gfn, self.charge, self.mult, self.optsteps)
            if self.solvmethod is not None :
                com_xtb += ' --alpb {}'.format(self.solvent)
            with open(self._path('result.out'), 'w') as out:
                xtb = subprocess.run(com_xtb, stdout=out, stderr=subprocess.STDOUT,
                                     shell=True, cwd=self.workdir)
            if xtb.returncode != 0:
                raise RuntimeError("xTB optimization failed with return code {}".format(xtb.returncode))

            with open(self._path('result.out'), 'r') as f:
                searchfile = f.readlines()

            opt_conv_key = "GEOMETRY OPTIMIZATION CONVERGED"

            not_conv = True
            for line in searchfile:
                if opt_conv_key in line:
                    not_conv = False

            #if not_conv:
            #    killfile = open("kill.dat", "w")
            #    killfile.close()

            if not_conv:
                raise RuntimeError("xTB geometry optimization did not converge")

            energy_key = "TOTAL ENERGY"
            energy = None
            for line in searchfile:
                if energy_key in line:
                    energy = float(line.split()[3])
            if energy is None:
                raise RuntimeError("xTB output did not contain TOTAL ENERGY")
            self.energy = energy

            if not os.path.exists(self._path('xtbopt.xyz')):
                raise RuntimeError("xTB optimization did not produce xtbopt.xyz")
            with open(self._path('xtbopt.xyz'), 'r') as f:
                self.xyz_string_opt = f.read()
                self.sdf_string_opt = laqa_fafoom.utilities.xyz2sdf(self.xyz_string_opt,
                                                                    self.sdf_string)
                success = True

        return success

    def get_energy(self, unit='eV'):
        """Get the energy of the molecule.

        Returns:
            energy (float) in eV, Hartree, or kcal/mol
        Raises:
            AttributeError: if energy hasn't been calculated yet
        """
        if not hasattr(self, 'energy'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            if unit == 'hartree':
                return self.energy
            elif unit == 'kcal':
                return hartree2kcalmol * self.energy
            else: # unit == 'eV':
                return hartree2eV * self.energy

    def get_gradient(self):
        """Get the gradient of the molecule.
        Returns:
            gradient (float) in [Hartree/bohr]
        Raises:
            AttributeError: if energy hasn't been calculated yet
        """
        if not hasattr(self, 'gradient'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.gradient

    def get_xyz_string_opt(self):
        """Get the optimized XYZ string produced by xTB.

        Returns:
            optimized xyz string (str)
        Raises:
            AttributeError: if the optimization hasn't been performed yet
        """
        if not hasattr(self, 'xyz_string_opt'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.xyz_string_opt

    def get_sdf_string_opt(self):
        """Get the optimized structure converted to an SDF string.

        Returns:
            optimized sdf string (str)
        Raises:
            AttributeError: if the optimization hasn't been performed yet
        """
        if not hasattr(self, 'sdf_string_opt'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.sdf_string_opt

    def save_to_file(self): 
        if os.path.isfile(self.sdf_out):
            f = open(self.sdf_out, "a")
        else:
            f = open(self.sdf_out, "w")
        s = str(self.sdf_string_opt) + '\n' \
            + ">  <Energy>"+'\n' \
            + str(self.energy)+'\n\n' \
            + "$$$$"+'\n'
        f.write(s)
        f.close()

    def clean(self):
        """Clean the working direction after the xtb calculation has been
        completed.
        """
        if self._owns_workdir:
            if self.workdir is not None:
                shutil.rmtree(self.workdir, ignore_errors=True)
                self.workdir = None
            return

        self._ensure_workdir()
        for f in glob.glob(os.path.join(self.workdir, "xtb*")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.workdir, "gfn*")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.workdir, "wbo")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.workdir, "charges")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.workdir, "energy")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.workdir, "gradient")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.workdir, "result.out")):
            os.remove(f)


def xtb_exec(sdf_string, xtb_call, jobtype='opt', gfn='2',
             charge=0, mult=1, optsteps=100, 
             solvmethod=None, solvent='water',
             sdf_out='optimized_structures.sdf'):

    xtb_object = xTBObject(sdf_string, xtb_call, jobtype, 
                           gfn, charge, mult, optsteps,
                           solvmethod, solvent, sdf_out)
    try:
        xtb_object.clean()
        xtb_object.generate_input()
        xtb_object.run_xtb()
        if jobtype == 'gradient':
            unit = 'hartree'
            energy = xtb_object.get_energy(unit)
            grad = xtb_object.get_gradient()
            return energy, grad
        else:
            unit = 'hartree'
            energy = xtb_object.get_energy(unit)
            xyz_string_opt = xtb_object.get_xyz_string_opt()
            return energy, xyz_string_opt
    finally:
        xtb_object.clean()


if __name__ == '__main__':

    sdf_string = '\nH2O\n\n' + \
    '  3  2  0  0  0  0  0  0  0  0999 V2000\n' + \
    '    0.0000    0.0000   -0.3894 O   0  0  0  0  0  0  0  0  0  0  0  0\n' + \
    '    0.7630    0.0000    0.1947 H   0  0  0  0  0  0  0  0  0  0  0  0\n' + \
    '   -0.7630    0.0000    0.1947 H   0  0  0  0  0  0  0  0  0  0  0  0\n' + \
    '  1  2  1  0  0  0  0\n' + \
    '  1  3  1  0  0  0  0\n' + \
    'M  END\n$$$$'
    xtb_call = '~/anaconda3/bin/xtb'
    gfn = '2' #'1', '2', 'ff'
    charge = 0
    mult = 1
    optsteps = 50
    solvmethod = None # 'alpb'
    solvent = None    # 'water'

    print('initial coord')
    print(sdf_string)

    jobtype='gradient'
    energy, grad = xtb_exec(sdf_string, xtb_call, jobtype,
                            gfn, charge, mult, optsteps, solvmethod, solvent)
                            
    print('energy:', energy)
    print('grad')
    for i in range(len(grad)):
        print(grad[i])

    jobtype='opt'
    energy, xyz_string_opt = xtb_exec(sdf_string, xtb_call, jobtype,
                            gfn, charge, mult, optsteps, solvmethod, solvent)

    print('energy:', energy)
    print('opt coord')
    print(xyz_string_opt)
