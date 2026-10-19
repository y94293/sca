"""Compare all five published SCA starts with the archived numerical outputs."""
from pathlib import Path
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTest(unittest.TestCase):
    def test_all_five_initializations_match_archived_sca_results(self):
        runner = ROOT / 'run_sca.py'
        self.assertTrue(runner.is_file(), 'The independent SCA runner is missing')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'results'
            process = subprocess.run(
                [sys.executable, str(runner), '--all-initials', '--verify',
                 '--output-dir', str(out)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            actual = json.loads((out / 'summary.json').read_text())['results']
            with (ROOT / 'reference/sca_expected.csv').open(newline='') as stream:
                reference = {int(row['offset_percent']): row for row in csv.DictReader(stream)}
            self.assertEqual(len(actual), 5)
            for result in actual:
                ref = reference[result['offset_percent']]
                best = result['best']
                for got, key in zip(best['theta'], ['final_aC_m', 'final_H_m']):
                    self.assertTrue(math.isclose(got, float(ref[key]), abs_tol=1e-14, rel_tol=0))
                for key, old in [('x0', 'final_x0_m'), ('J', 'J'),
                                 ('coarse_area_mm2', 'fixed_grid_area_mm2'),
                                 ('M0', 'nominal_M_H')]:
                    self.assertTrue(math.isclose(best[key], float(ref[old]), abs_tol=1e-12, rel_tol=0), key)
                self.assertEqual(result['status'], ref['termination'])
                self.assertTrue(result['verified'])


if __name__ == '__main__':
    unittest.main()
