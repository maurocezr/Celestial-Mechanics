import unittest

import numpy as np

from nbody import psr_j1757_1854


class PSRJ17571854PresetTests(unittest.TestCase):
    def test_psr_j1757_1854_preset_returns_expected_shapes_and_scale(self):
        positions, velocities, masses, G = psr_j1757_1854()
        self.assertEqual(positions.shape, (2, 3))
        self.assertEqual(velocities.shape, (2, 3))
        self.assertEqual(masses.shape, (2,))
        self.assertAlmostEqual(G, 4.0 * np.pi * np.pi)
        self.assertGreater(float(np.linalg.norm(positions[1] - positions[0])), 0.0)
        self.assertGreater(float(np.linalg.norm(velocities[1] - velocities[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
