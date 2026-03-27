import unittest

from analysis import arcsec_per_century_from_rad_per_year, estimate_precession_rate
from nbody import SimulationConfig, inner_solar_system_toy, run_simulation


class CentralBodyMultibodyTests(unittest.TestCase):
    def test_central_body_multibody_run_produces_nonzero_inner_precession_shift(self):
        years = 3.0
        dt = 2e-4
        save_every = 2
        steps = int(round(years / dt))

        positions, velocities, masses, G = inner_solar_system_toy()
        cfg_newtonian = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="inner-solar-system-toy",
        )
        res_newtonian = run_simulation(positions, velocities, masses, G, cfg_newtonian)

        positions, velocities, masses, G = inner_solar_system_toy()
        cfg_central = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="inner-solar-system-toy",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="central-body",
            pn_primary_index=0,
        )
        res_central = run_simulation(positions, velocities, masses, G, cfg_central)

        baseline_subtracted_rate = estimate_precession_rate(
            res_central["times"], res_central["traj"], primary_index=0, secondary_index=1
        ) - estimate_precession_rate(
            res_newtonian["times"], res_newtonian["traj"], primary_index=0, secondary_index=1
        )
        arcsec_per_century = arcsec_per_century_from_rad_per_year(baseline_subtracted_rate)

        self.assertGreater(abs(arcsec_per_century), 10.0)


if __name__ == "__main__":
    unittest.main()
