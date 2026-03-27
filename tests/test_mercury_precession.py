import unittest

from analysis import arcsec_per_century_from_rad_per_year, estimate_precession_rate
from nbody import SimulationConfig, mercury_relativistic_demo, run_simulation


class MercuryPrecessionTests(unittest.TestCase):
    def test_mercury_demo_shows_nonzero_precession(self):
        years = 5.0
        dt = 2e-4
        save_every = 2
        steps = int(round(years / dt))

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_newtonian = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
        )
        res_newtonian = run_simulation(positions, velocities, masses, G, cfg_newtonian)

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_1pn = SimulationConfig(
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
        res_1pn = run_simulation(positions, velocities, masses, G, cfg_1pn)

        baseline_subtracted_rate = estimate_precession_rate(res_1pn["times"], res_1pn["traj"]) - estimate_precession_rate(
            res_newtonian["times"], res_newtonian["traj"]
        )
        arcsec_per_century = arcsec_per_century_from_rad_per_year(baseline_subtracted_rate)

        self.assertGreater(arcsec_per_century, 40.0)
        self.assertLess(arcsec_per_century, 46.0)


if __name__ == "__main__":
    unittest.main()
