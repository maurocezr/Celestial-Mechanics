import unittest
import math

import numpy as np

from analysis import arcsec_per_century_from_rad_per_year, estimate_precession_rate
from nbody import SimulationConfig, binary_pulsar_toy, psr_b1913_16, psr_j1757_1854, run_simulation
from relativity import (
    _two_body_harmonic_2pn_acceleration,
    combined_1pn_acceleration,
    combined_2p5pn_acceleration,
    post_newtonian_2p5pn_acceleration,
)


class BinaryPulsarToyTests(unittest.TestCase):
    def test_hulse_taylor_2p5pn_matches_famous_orbital_decay_scale(self):
        positions, velocities, masses, G = psr_b1913_16()
        m1, m2 = float(masses[0]), float(masses[1])
        total_mass = m1 + m2
        rel_pos = positions[1] - positions[0]
        rel_vel = velocities[1] - velocities[0]
        radius = float(np.linalg.norm(rel_pos))
        speed2 = float(np.dot(rel_vel, rel_vel))
        semi_major_axis = 1.0 / (2.0 / radius - speed2 / (G * total_mass))
        angular_momentum = np.cross(rel_pos, rel_vel)
        eccentricity_vector = np.cross(rel_vel, angular_momentum) / (G * total_mass) - rel_pos / radius
        eccentricity = float(np.linalg.norm(eccentricity_vector))
        orbital_period_years = math.sqrt(semi_major_axis**3 / total_mass)
        sqrt_one_minus_e2 = math.sqrt(1.0 - eccentricity * eccentricity)

        mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        eccentric_anomaly = mean_anomaly.copy()
        for _ in range(12):
            eccentric_anomaly -= (
                eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
            ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))

        cos_e = np.cos(eccentric_anomaly)
        sin_e = np.sin(eccentric_anomaly)
        rel_positions = np.column_stack(
            (
                semi_major_axis * (cos_e - eccentricity),
                semi_major_axis * sqrt_one_minus_e2 * sin_e,
                np.zeros_like(eccentric_anomaly),
            )
        )
        rel_radii = semi_major_axis * (1.0 - eccentricity * cos_e)
        rel_velocities = np.column_stack(
            (
                -np.sqrt(G * total_mass * semi_major_axis) * sin_e / rel_radii,
                np.sqrt(G * total_mass * semi_major_axis) * sqrt_one_minus_e2 * cos_e / rel_radii,
                np.zeros_like(eccentric_anomaly),
            )
        )

        dissipative_power = np.empty(mean_anomaly.size, dtype=np.float64)
        for i, (rel_position_i, rel_velocity_i) in enumerate(zip(rel_positions, rel_velocities)):
            sample_positions = np.asarray(
                [
                    -(m2 / total_mass) * rel_position_i,
                    (m1 / total_mass) * rel_position_i,
                ],
                dtype=np.float64,
            )
            sample_velocities = np.asarray(
                [
                    -(m2 / total_mass) * rel_velocity_i,
                    (m1 / total_mass) * rel_velocity_i,
                ],
                dtype=np.float64,
            )
            sample_acceleration = post_newtonian_2p5pn_acceleration(
                sample_positions,
                sample_velocities,
                masses,
                G,
                1e-9,
                63239.7263,
                scope="two-body",
            )
            dissipative_power[i] = float(np.sum(masses[:, None] * sample_velocities * sample_acceleration))

        mean_dissipative_power = float(np.mean(dissipative_power))
        pb_dot_years_per_year = (3.0 * semi_major_axis / (G * m1 * m2)) * mean_dissipative_power * orbital_period_years
        pb_dot_microseconds_per_year = pb_dot_years_per_year * 365.25 * 24.0 * 3600.0 * 1e6

        # The Hulse-Taylor binary PSR B1913+16 is famously observed to decay by about
        # 75 microseconds per year. The implementation-backed 2.5PN quadrupole average
        # should reproduce that sign and scale.
        self.assertLess(pb_dot_microseconds_per_year, 0.0)
        self.assertAlmostEqual(pb_dot_microseconds_per_year, -75.0, delta=2.0)

    def test_standalone_1pn_force_regression_fixed_state(self):
        positions = np.asarray(
            [
                [-1.2e-3, 2.5e-4, 0.0],
                [9.0e-4, -1.5e-4, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [
                [18.0, -125.0, 0.0],
                [-22.0, 134.0, 0.0],
            ],
            dtype=np.float64,
        )
        masses = np.asarray([1.338185, 1.248868], dtype=np.float64)
        G = 4.0 * np.pi * np.pi
        c = 63239.7263
        eps = 1e-9

        actual = combined_1pn_acceleration(positions, velocities, masses, G, eps, c, scope="two-body")
        expected = np.asarray(
            [
                [10597643.334711567, -2018390.4112681171, 0.0],
                [-11355569.480410255, 2162742.3975174516, 0.0],
            ],
            dtype=np.float64,
        )

        self.assertTrue(np.allclose(actual, expected, rtol=1e-13, atol=1e-13))

    def test_harmonic_2pn_fixed_state_matches_dire_formula(self):
        positions = np.asarray(
            [
                [-1.2e-3, 2.5e-4, 0.0],
                [9.0e-4, -1.5e-4, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [
                [18.0, -125.0, 0.0],
                [-22.0, 134.0, 0.0],
            ],
            dtype=np.float64,
        )
        masses = np.asarray([1.338185, 1.248868], dtype=np.float64)
        G = 4.0 * np.pi * np.pi
        c = 63239.7263

        actual = _two_body_harmonic_2pn_acceleration(positions, velocities, masses, G, c)

        rel_pos = positions[1] - positions[0]
        rel_vel = velocities[1] - velocities[0]
        radius = float(np.linalg.norm(rel_pos))
        radial_velocity = float(np.dot(rel_pos, rel_vel)) / radius
        speed2 = float(np.dot(rel_vel, rel_vel))
        total_mass = float(np.sum(masses))
        eta = float(np.prod(masses) / (total_mass * total_mass))
        gm = G * total_mass
        gm_over_r = gm / radius

        a_2pn = (
            -eta * (3.0 - 4.0 * eta) * speed2 * speed2
            + 0.5 * eta * (13.0 - 4.0 * eta) * speed2 * gm_over_r
            + 1.5 * eta * (3.0 - 4.0 * eta) * speed2 * radial_velocity * radial_velocity
            + (2.0 + 25.0 * eta + 2.0 * eta * eta) * radial_velocity * radial_velocity * gm_over_r
            - 1.875 * eta * (1.0 - 3.0 * eta) * radial_velocity**4
            - 0.75 * (12.0 + 29.0 * eta) * gm_over_r * gm_over_r
        )
        b_2pn = (
            0.5 * eta * (15.0 + 4.0 * eta) * speed2
            - 1.5 * eta * (3.0 + 2.0 * eta) * radial_velocity * radial_velocity
            - 0.5 * (4.0 + 41.0 * eta + 8.0 * eta * eta) * gm_over_r
        )
        rel_expected = (gm / (c**4 * radius**3)) * (
            a_2pn * rel_pos + b_2pn * radial_velocity * radius * rel_vel
        )
        expected = np.asarray(
            [
                -(masses[1] / total_mass) * rel_expected,
                (masses[0] / total_mass) * rel_expected,
            ],
            dtype=np.float64,
        )

        self.assertTrue(np.allclose(actual, expected, rtol=1e-13, atol=1e-13))
        self.assertLess(float(np.linalg.norm(np.sum(masses[:, None] * actual, axis=0))), 1e-15)

    def test_upgraded_2p5pn_adds_harmonic_2pn_on_top_of_previous_baseline(self):
        positions = np.asarray(
            [
                [-1.4e-3, 3.0e-4, 0.0],
                [1.1e-3, -2.0e-4, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [
                [24.0, -141.0, 0.0],
                [-28.0, 153.0, 0.0],
            ],
            dtype=np.float64,
        )
        masses = np.asarray([1.338185, 1.248868], dtype=np.float64)
        G = 4.0 * np.pi * np.pi
        c = 63239.7263
        eps = 1e-9

        legacy_baseline = combined_1pn_acceleration(positions, velocities, masses, G, eps, c, scope="two-body")
        legacy_baseline += post_newtonian_2p5pn_acceleration(positions, velocities, masses, G, eps, c, scope="two-body")

        harmonic_2pn = _two_body_harmonic_2pn_acceleration(positions, velocities, masses, G, c)
        upgraded = combined_2p5pn_acceleration(positions, velocities, masses, G, eps, c, scope="two-body")

        self.assertTrue(np.allclose(upgraded, legacy_baseline + harmonic_2pn, rtol=1e-13, atol=1e-13))
        self.assertGreater(float(np.max(np.abs(harmonic_2pn))), 0.0)

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

    def test_2p5pn_compact_binary_loses_bookkeeping_energy(self):
        years = 0.01
        dt = 1e-6
        steps = int(round(years / dt))

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
        res_2p5pn = run_simulation(positions, velocities, masses, G, cfg_2p5pn, record_energy=True)

        self.assertLess(float(res_2p5pn["energy"][-1]), float(res_2p5pn["energy"][0]) - 1e-3)


if __name__ == "__main__":
    unittest.main()
