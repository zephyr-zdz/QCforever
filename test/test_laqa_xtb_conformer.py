import os
import sys
import tempfile
import textwrap
import types
import unittest

from rdkit import Chem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.modules.setdefault(
    "psutil",
    types.SimpleNamespace(cpu_count=lambda: 1, cpu_percent=lambda interval=None: 0),
)
scipy_stub = types.ModuleType("scipy")
scipy_stats_stub = types.ModuleType("scipy.stats")
scipy_signal_stub = types.ModuleType("scipy.signal")
scipy_integrate_stub = types.ModuleType("scipy.integrate")
scipy_stats_stub.wasserstein_distance = lambda *args, **kwargs: 0
scipy_signal_stub.correlate = lambda *args, **kwargs: []
scipy_integrate_stub.simpson = lambda *args, **kwargs: 0
scipy_stub.stats = scipy_stats_stub
scipy_stub.signal = scipy_signal_stub
scipy_stub.integrate = scipy_integrate_stub
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.stats", scipy_stats_stub)
sys.modules.setdefault("scipy.signal", scipy_signal_stub)
sys.modules.setdefault("scipy.integrate", scipy_integrate_stub)

from qcforever import laqa_fafoom


WATER_SDF = """water
  QCforever

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000   -0.3894 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7630    0.0000    0.1947 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7630    0.0000    0.1947 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
"""


class LAQAXTBConformerTests(unittest.TestCase):
    def test_make_laqa_input_accepts_uppercase_xtb_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                laqa_fafoom.laqa_confopt_QCforever.make_laqa_input(
                    "CCO", 1, 0, 1, "XTB", 1, "")

                with open("laqa_setting.inp") as f:
                    settings = f.read()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(settings.count('energy_function = "xtb"'), 2)
        self.assertIn("mult = 1", settings)
        self.assertIn("charge = 0", settings)

    def test_xtb_uses_unpaired_electrons_for_uhf_argument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_xtb = os.path.join(tmpdir, "mock_xtb.py")
            args_file = os.path.join(tmpdir, "args.txt")
            with open(mock_xtb, "w") as f:
                f.write(textwrap.dedent("""
                    import os
                    import shutil
                    import sys
                    with open(os.environ["MOCK_XTB_ARGS"], "a") as args:
                        args.write(" ".join(sys.argv[1:]) + "\\n")
                    shutil.copyfile("xtbin.xyz", "xtbopt.xyz")
                    print("GEOMETRY OPTIMIZATION CONVERGED")
                    print(" | TOTAL ENERGY      -1.234 Eh")
                """))

            old_args_file = os.environ.get("MOCK_XTB_ARGS")
            os.environ["MOCK_XTB_ARGS"] = args_file
            try:
                xtb_call = '{} "{}"'.format(sys.executable, mock_xtb)
                for mult in (1, 2, 3):
                    laqa_fafoom.pyxtb.xtb_exec(
                        WATER_SDF, xtb_call, jobtype='opt', mult=mult)
            finally:
                if old_args_file is None:
                    os.environ.pop("MOCK_XTB_ARGS", None)
                else:
                    os.environ["MOCK_XTB_ARGS"] = old_args_file

            with open(args_file) as f:
                args_lines = f.read().splitlines()

            self.assertIn("--uhf 0", args_lines[0])
            self.assertIn("--uhf 1", args_lines[1])
            self.assertIn("--uhf 2", args_lines[2])

    def test_xtb_opt_uses_isolated_workdir_and_returns_xyz_convertible_to_sdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_xtb = os.path.join(tmpdir, "mock_xtb.py")
            with open(mock_xtb, "w") as f:
                f.write(textwrap.dedent("""
                    import shutil
                    shutil.copyfile("xtbin.xyz", "xtbopt.xyz")
                    print("GEOMETRY OPTIMIZATION CONVERGED")
                    print(" | TOTAL ENERGY      -1.234 Eh")
                """))

            cwd = os.getcwd()
            xtb_call = '{} "{}"'.format(sys.executable, mock_xtb)
            energy, xyz_string = laqa_fafoom.pyxtb.xtb_exec(
                WATER_SDF, xtb_call, jobtype='opt')

            sdf_string = laqa_fafoom.utilities.xyz2sdf(xyz_string, WATER_SDF)
            mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)

            self.assertAlmostEqual(energy, -1.234)
            self.assertIsNotNone(mol)
            self.assertFalse(os.path.exists(os.path.join(cwd, "xtbin.xyz")))
            self.assertFalse(os.path.exists(os.path.join(cwd, "xtbopt.xyz")))

    def test_generate_conformers_batch_returns_valid_sdf_records(self):
        conformers = laqa_fafoom.get_parameters.generate_conformers_batch(
            "CCO", 3, 1.3, 2.15, overgenerate_factor=1.0,
            rmsd_prune=0.1, seed=1)

        self.assertGreater(len(conformers), 0)
        self.assertLessEqual(len(conformers), 3)
        for sdf_string in conformers:
            self.assertIsNotNone(Chem.MolFromMolBlock(sdf_string, removeHs=False))
            self.assertTrue(laqa_fafoom.utilities.check_geo_sdf(sdf_string, 1.3, 2.15))

    def test_initgeom_falls_back_to_legacy_when_batch_returns_no_candidates(self):
        original = laqa_fafoom.get_parameters.generate_conformers_batch
        laqa_fafoom.get_parameters.generate_conformers_batch = (
            lambda *args, **kwargs: [])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    with open("laqa_setting.inp", "w") as f:
                        f.write(textwrap.dedent("""
                            [Molecule]
                            smiles = "CCO"

                            [Initial geometry]
                            popsize = 2
                            cnt_max = 30
                            energy_function = "ff"
                            initgeom_method = "batch_etkdg"
                            overgenerate_factor = 1.0
                            rmsd_prune = 0.1
                            seed = 1
                            sdf_out = "initial_structures.sdf"
                        """))

                    laqa_fafoom.initgeom.LAQA_initgeom("laqa_setting.inp")
                    mols = [
                        mol for mol in Chem.SDMolSupplier(
                            "initial_structures.sdf", removeHs=False)
                        if mol is not None
                    ]

                    self.assertGreater(len(mols), 0)
                    self.assertLessEqual(len(mols), 2)
                finally:
                    os.chdir(old_cwd)
        finally:
            laqa_fafoom.get_parameters.generate_conformers_batch = original


if __name__ == "__main__":
    unittest.main()
