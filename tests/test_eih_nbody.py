import unittest

import numpy as np

from analysis import arcsec_per_century_from_rad_per_year, estimate_precession_rate
from nbody import SimulationConfig, mercury_relativistic_demo, run_simulation, star_earth_jupiter
from relativity import eih_energy_components, total_energy_1pn


class EIHNBodyTests(unittest.TestCase):
    def test_eih_converges_to_newtonian_for_large_c_in_multibody_case(self):
        years = 2.0
        dt = 5e-4
        steps = int(round(years / dt))

        positions, velocities, masses, G = star_earth_jupiter()
        cfg_newtonian = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="star-earth-jupiter",
        )
        res_newtonian = run_simulation(positions, velocities, masses, G, cfg_newtonian)

        positions, velocities, masses, G = star_earth_jupiter()
        cfg_eih = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="star-earth-jupiter",
            gravity_model="1pn",
            c=1e8,
            pn_scope="eih",
        )
        res_eih = run_simulation(positions, velocities, masses, G, cfg_eih)

        self.assertLess(float(np.max(np.abs(res_eih["traj"] - res_newtonian["traj"]))), 1e-8)
        self.assertLess(float(np.max(np.abs(res_eih["vel"] - res_newtonian["vel"]))), 1e-8)

    def test_eih_matches_two_body_scope_for_mercury_case(self):
        years = 5.0
        dt = 2e-4
        steps = int(round(years / dt))

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_two_body = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=2,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_two_body = run_simulation(positions, velocities, masses, G, cfg_two_body)

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_eih = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=2,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="eih",
        )
        res_eih = run_simulation(positions, velocities, masses, G, cfg_eih)

        rate_two_body = estimate_precession_rate(res_two_body["times"], res_two_body["traj"])
        rate_eih = estimate_precession_rate(res_eih["times"], res_eih["traj"])
        diff_arcsec = arcsec_per_century_from_rad_per_year(rate_eih - rate_two_body)
        self.assertLess(abs(diff_arcsec), 1.0)

    def test_eih_energy_components_match_total_energy(self):
        positions, velocities, masses, G = star_earth_jupiter()
        K, U, E = eih_energy_components(positions, velocities, masses, G, 63239.7263)
        total = total_energy_1pn(positions, velocities, masses, G, 63239.7263, scope="eih")
        self.assertAlmostEqual(K + U, E, places=12)
        self.assertAlmostEqual(E, total, places=12)

    def test_run_simulation_marks_eih_energy_kind(self):
        positions, velocities, masses, G = star_earth_jupiter()
        cfg_eih = SimulationConfig(
            dt=0.001,
            steps=10,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="star-earth-jupiter",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="eih",
        )
        res = run_simulation(positions, velocities, masses, G, cfg_eih)
        self.assertEqual(res["energy_kind"], "1pn-eih")


if __name__ == "__main__":
    unittest.main()
