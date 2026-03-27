import unittest

import numpy as np

from nbody import SimulationConfig, random_nbody, run_simulation, two_body_sun_earth


class NewtonianRegressionTests(unittest.TestCase):
    def test_newtonian_two_body_matches_baseline(self):
        positions, velocities, masses, G = two_body_sun_earth()
        cfg = SimulationConfig(
            dt=0.001,
            steps=1000,
            save_every=100,
            eps=1e-3,
            integrator="leapfrog",
            preset="two-body",
        )
        res = run_simulation(positions, velocities, masses, G, cfg)

        expected_final_pos = np.array(
            [
                [-3.003480587882155e-06, 1.915872434574192e-10, 0.0],
                [9.999969944849308e-01, -6.378821569942659e-05, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(res["traj"][-1], expected_final_pos, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(res["energy"][-1]), -5.928662742340585e-05, places=15)

    def test_newtonian_random_matches_baseline(self):
        positions, velocities, masses, G = random_nbody(8, seed=7, mass_spread=0.2)
        cfg = SimulationConfig(
            dt=0.002,
            steps=200,
            save_every=20,
            eps=1e-3,
            integrator="rk4",
            preset="random",
        )
        res = run_simulation(positions, velocities, masses, G, cfg)

        expected_final_pos = np.array(
            [
                [10.412374860763736, 20.617083293377757, 2.404898948017352],
                [6.73021385351287, -27.61985924907896, 21.478986735372242],
                [-9.896612568048774, -19.127926772545788, -2.332616821023876],
                [-0.335174765204884, 0.573258331003368, 0.464956465841608],
                [2.398216943568595, -4.151735876084126, -1.812834659337207],
                [-2.355252303440875, 4.006237498921478, 2.094117905180839],
                [-6.855681012094836, 25.750947235199618, -20.86050073807483],
                [0.66182290096053, -1.214040992330139, -0.150345311487341],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(res["traj"][-1], expected_final_pos, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(res["energy"][-1]), 43654.559865466596, places=9)


if __name__ == "__main__":
    unittest.main()
