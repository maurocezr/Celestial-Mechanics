import unittest

import numpy as np

from analysis import arcsec_per_century_from_rad_per_year, estimate_precession_rate
from nbody import SimulationConfig, binary_pulsar_toy, psr_j1757_1854, run_simulation


class BinaryPulsarToyTests(unittest.TestCase):
    def test_binary_demo_differs_from_newtonian(self):
        years = 0.05
        dt = 1e-5
        save_every = 10
        steps = int(round(years / dt))

        positions, velocities, masses, G = binary_pulsar_toy()
        cfg_newtonian = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="binary-pulsar-toy",
        )
        res_newtonian = run_simulation(positions, velocities, masses, G, cfg_newtonian)

        positions, velocities, masses, G = binary_pulsar_toy()
        cfg_1pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=save_every,
            eps=1e-3,
            integrator="rk4",
            preset="binary-pulsar-toy",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_1pn = run_simulation(positions, velocities, masses, G, cfg_1pn)

        baseline_subtracted_rate = estimate_precession_rate(res_1pn["times"], res_1pn["traj"]) - estimate_precession_rate(
            res_newtonian["times"], res_newtonian["traj"]
        )
        arcsec_per_century = arcsec_per_century_from_rad_per_year(baseline_subtracted_rate)

        self.assertGreater(abs(arcsec_per_century), 1e5)

    def test_psr_j1757_1854_shows_noticeable_2pn_shift_against_1pn(self):
        years = 0.005
        dt = 1e-6
        steps = int(round(years / dt))

        positions, velocities, masses, G = psr_j1757_1854()
        cfg_1pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=1,
            eps=1e-9,
            integrator="rk4",
            preset="psr-j1757-1854",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_1pn = run_simulation(positions, velocities, masses, G, cfg_1pn, record_energy=False)

        positions, velocities, masses, G = psr_j1757_1854()
        cfg_2pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=1,
            eps=1e-9,
            integrator="rk4",
            preset="psr-j1757-1854",
            gravity_model="2pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_2pn = run_simulation(positions, velocities, masses, G, cfg_2pn, record_energy=False)

        rate_1pn = estimate_precession_rate(res_1pn["times"], res_1pn["traj"])
        rate_2pn = estimate_precession_rate(res_2pn["times"], res_2pn["traj"])
        delta_precession_arcsec_per_century = arcsec_per_century_from_rad_per_year(rate_2pn - rate_1pn)
        max_position_delta = float(np.max(np.linalg.norm(res_2pn["traj"] - res_1pn["traj"], axis=2)))

        self.assertGreater(abs(delta_precession_arcsec_per_century), 10.0)
        self.assertGreater(max_position_delta, 1.0e-10)

    def test_2p5pn_compact_binary_differs_measurably_from_1pn(self):
        years = 0.03
        dt = 1e-6
        steps = int(round(years / dt))

        positions, velocities, masses, G = psr_j1757_1854()
        cfg_1pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=10,
            eps=1e-9,
            integrator="rk4",
            preset="psr-j1757-1854",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_1pn = run_simulation(positions, velocities, masses, G, cfg_1pn, record_energy=False)

        positions, velocities, masses, G = psr_j1757_1854()
        cfg_2p5pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=10,
            eps=1e-9,
            integrator="rk4",
            preset="psr-j1757-1854",
            gravity_model="2.5pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        res_2p5pn = run_simulation(positions, velocities, masses, G, cfg_2p5pn, record_energy=False)

        max_position_delta = float(np.max(np.linalg.norm(res_2p5pn["traj"] - res_1pn["traj"], axis=2)))
        max_velocity_delta = float(np.max(np.linalg.norm(res_2p5pn["vel"] - res_1pn["vel"], axis=2)))

        self.assertGreater(max_position_delta, 1e-10)
        self.assertGreater(max_velocity_delta, 1e-6)

    def test_2p5pn_converges_to_1pn_for_large_c(self):
        years = 0.01
        dt = 1e-5
        steps = int(round(years / dt))

        positions, velocities, masses, G = binary_pulsar_toy()
        cfg_1pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="binary-pulsar-toy",
            gravity_model="1pn",
            c=1e10,
            pn_scope="two-body",
        )
        res_1pn = run_simulation(positions, velocities, masses, G, cfg_1pn, record_energy=False)

        positions, velocities, masses, G = binary_pulsar_toy()
        cfg_2p5pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=5,
            eps=1e-3,
            integrator="rk4",
            preset="binary-pulsar-toy",
            gravity_model="2.5pn",
            c=1e10,
            pn_scope="two-body",
        )
        res_2p5pn = run_simulation(positions, velocities, masses, G, cfg_2p5pn, record_energy=False)

        self.assertLess(float(np.max(np.abs(res_2p5pn["traj"] - res_1pn["traj"]))), 1e-8)
        self.assertLess(float(np.max(np.abs(res_2p5pn["vel"] - res_1pn["vel"]))), 1e-8)


if __name__ == "__main__":
    unittest.main()
