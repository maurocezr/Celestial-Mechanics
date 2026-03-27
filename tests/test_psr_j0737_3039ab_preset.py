import unittest

import numpy as np

from nbody import psr_j0737_3039ab


class PSRJ07373039ABPresetTests(unittest.TestCase):
    def test_psr_j0737_3039ab_preset_returns_expected_shapes_and_scale(self):
        positions, velocities, masses, G = psr_j0737_3039ab()
        self.assertEqual(positions.shape, (2, 3))
        self.assertEqual(velocities.shape, (2, 3))
        self.assertEqual(masses.shape, (2,))
        self.assertAlmostEqual(G, 4.0 * np.pi * np.pi)
        self.assertGreater(float(np.linalg.norm(positions[1] - positions[0])), 0.0)
        self.assertGreater(float(np.linalg.norm(velocities[1] - velocities[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
