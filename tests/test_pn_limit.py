import unittest

import numpy as np

from nbody import SimulationConfig, mercury_relativistic_demo, run_simulation


class PostNewtonianLimitTests(unittest.TestCase):
    def test_1pn_converges_to_newtonian_for_large_c(self):
        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_newtonian = SimulationConfig(
            dt=5e-4,
            steps=int(round(2.0 / 5e-4)),
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
        )
        res_newtonian = run_simulation(positions, velocities, masses, G, cfg_newtonian)

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_1pn = SimulationConfig(
            dt=5e-4,
            steps=int(round(2.0 / 5e-4)),
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=1e8,
            pn_scope="two-body",
        )
        res_1pn = run_simulation(positions, velocities, masses, G, cfg_1pn)

        max_traj_diff = float(np.max(np.abs(res_1pn["traj"] - res_newtonian["traj"])))
        max_vel_diff = float(np.max(np.abs(res_1pn["vel"] - res_newtonian["vel"])))

        self.assertLess(max_traj_diff, 1e-9)
        self.assertLess(max_vel_diff, 1e-9)


if __name__ == "__main__":
    unittest.main()
