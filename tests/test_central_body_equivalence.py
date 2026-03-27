import unittest

import numpy as np

from analysis import arcsec_per_century_from_rad_per_year, estimate_precession_rate
from nbody import SimulationConfig, mercury_relativistic_demo, run_simulation


class CentralBodyEquivalenceTests(unittest.TestCase):
    def test_central_body_matches_two_body_for_mercury_case(self):
        years = 5.0
        dt = 2e-4
        save_every = 2
        steps = int(round(years / dt))

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_two_body = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_two_body = run_simulation(positions, velocities, masses, G, cfg_two_body)

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_central = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="central-body",
            pn_primary_index=0,
        )
        res_central = run_simulation(positions, velocities, masses, G, cfg_central)

        np.testing.assert_allclose(res_two_body["traj"], res_central["traj"], rtol=0.0, atol=1e-10)

        rate_two_body = estimate_precession_rate(res_two_body["times"], res_two_body["traj"])
        rate_central = estimate_precession_rate(res_central["times"], res_central["traj"])
        diff_arcsec = arcsec_per_century_from_rad_per_year(rate_central - rate_two_body)
        self.assertLess(abs(diff_arcsec), 1.0)

    def test_central_body_converges_to_newtonian_for_large_c(self):
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
        cfg_central = SimulationConfig(
            dt=5e-4,
            steps=int(round(2.0 / 5e-4)),
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=1e8,
            pn_scope="central-body",
            pn_primary_index=0,
        )
        res_central = run_simulation(positions, velocities, masses, G, cfg_central)

        self.assertLess(float(np.max(np.abs(res_central["traj"] - res_newtonian["traj"]))), 1e-9)
        self.assertLess(float(np.max(np.abs(res_central["vel"] - res_newtonian["vel"]))), 1e-9)


if __name__ == "__main__":
    unittest.main()
