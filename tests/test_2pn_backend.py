import unittest

import numpy as np

from nbody import (
    SimulationConfig,
    canonical_momenta_from_velocities,
    mercury_relativistic_demo,
    run_canonical_simulation,
    run_simulation,
    star_earth_jupiter,
    newtonian_canonical_rhs,
)
from relativity import (
    solve_two_body_adm_2pn_momentum_from_velocity,
    two_body_adm_2pn_reduced_hamiltonian,
    two_body_adm_2pn_rhs,
)


class TwoPNBackendTests(unittest.TestCase):
    def test_analytic_2pn_rhs_matches_finite_difference_hamiltonian_gradients(self):
        masses = np.asarray([1.338185, 1.248868], dtype=np.float64)
        G = 4.0 * np.pi * np.pi
        c = 63239.7263
        reduced_mass = float(np.prod(masses) / np.sum(masses))
        rng = np.random.default_rng(12345)

        for _ in range(5):
            relative_position = np.asarray(
                [
                    rng.uniform(1.0e-3, 8.0e-3),
                    rng.uniform(-2.0e-3, 2.0e-3),
                    rng.uniform(-1.0e-5, 1.0e-5),
                ],
                dtype=np.float64,
            )
            relative_momentum = np.asarray(
                [
                    rng.uniform(-40.0, 40.0),
                    rng.uniform(80.0, 180.0),
                    rng.uniform(-1.0e-2, 1.0e-2),
                ],
                dtype=np.float64,
            )

            analytic_dr_dt, analytic_dp_dt = two_body_adm_2pn_rhs(
                relative_position,
                relative_momentum,
                masses,
                G,
                c,
            )

            fd_dh_dp = np.empty(3, dtype=np.float64)
            for axis in range(3):
                step = 1e-8 * max(1.0, abs(relative_momentum[axis]))
                plus = relative_momentum.copy()
                minus = relative_momentum.copy()
                plus[axis] += step
                minus[axis] -= step
                fd_dh_dp[axis] = (
                    two_body_adm_2pn_reduced_hamiltonian(relative_position, plus, masses, G, c)
                    - two_body_adm_2pn_reduced_hamiltonian(relative_position, minus, masses, G, c)
                ) / (2.0 * step)

            fd_dh_dr = np.empty(3, dtype=np.float64)
            for axis in range(3):
                step = 1e-8 * max(1.0, abs(relative_position[axis]))
                plus = relative_position.copy()
                minus = relative_position.copy()
                plus[axis] += step
                minus[axis] -= step
                fd_dh_dr[axis] = (
                    two_body_adm_2pn_reduced_hamiltonian(plus, relative_momentum, masses, G, c)
                    - two_body_adm_2pn_reduced_hamiltonian(minus, relative_momentum, masses, G, c)
                ) / (2.0 * step)

            self.assertTrue(np.allclose(analytic_dr_dt, reduced_mass * fd_dh_dp, rtol=1e-6, atol=1e-4))
            self.assertTrue(np.allclose(analytic_dp_dt, -reduced_mass * fd_dh_dr, rtol=1e-6, atol=5e-2))

    def test_canonical_newtonian_scaffold_matches_existing_newtonian_path(self):
        years = 0.05
        dt = 0.001
        steps = int(round(years / dt))

        positions, velocities, masses, G = star_earth_jupiter()
        cfg = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="star-earth-jupiter",
        )
        direct = run_simulation(positions, velocities, masses, G, cfg)

        positions, velocities, masses, G = star_earth_jupiter()
        momenta = canonical_momenta_from_velocities(velocities, masses)
        canonical = run_canonical_simulation(
            positions,
            momenta,
            masses,
            G,
            cfg,
            rhs_fn=newtonian_canonical_rhs,
        )

        self.assertLess(float(np.max(np.abs(canonical["traj"] - direct["traj"]))), 1e-10)
        self.assertLess(float(np.max(np.abs(canonical["vel"] - direct["vel"]))), 1e-10)

    def test_existing_1pn_force_path_still_runs(self):
        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg = SimulationConfig(
            dt=0.001,
            steps=5,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        results = run_simulation(positions, velocities, masses, G, cfg)
        self.assertEqual(results["energy_kind"], "1pn")

    def test_2pn_two_body_runs_with_canonical_backend(self):
        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg = SimulationConfig(
            dt=0.001,
            steps=5,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="2pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        results = run_simulation(positions, velocities, masses, G, cfg)
        self.assertEqual(results["energy_kind"], "2pn-two-body")
        self.assertEqual(results["traj"].shape[1], 2)
        self.assertIn("pn_primary_index", results)
        self.assertIn("c", results)
        self.assertIn("body_names", results)

    def test_2pn_solver_matches_requested_initial_relative_velocity(self):
        positions, velocities, masses, G = mercury_relativistic_demo()
        relative_position = positions[0] - positions[1]
        relative_velocity = velocities[0] - velocities[1]
        momentum = solve_two_body_adm_2pn_momentum_from_velocity(
            relative_position,
            relative_velocity,
            masses,
            G,
            63239.7263,
        )
        solved_velocity, _ = two_body_adm_2pn_rhs(
            relative_position,
            momentum,
            masses,
            G,
            63239.7263,
        )
        self.assertLess(float(np.linalg.norm(solved_velocity - relative_velocity)), 1e-11)

    def test_2pn_run_preserves_requested_initial_velocity(self):
        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg = SimulationConfig(
            dt=0.001,
            steps=2,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="2pn",
            c=63239.7263,
            pn_scope="two-body",
        )
        results = run_simulation(positions, velocities, masses, G, cfg)
        self.assertLess(float(np.max(np.abs(results["vel"][0] - velocities))), 1e-11)

    def test_2pn_two_body_converges_to_1pn_for_large_c(self):
        years = 0.1
        dt = 0.001
        steps = int(round(years / dt))

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_1pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="1pn",
            c=1e8,
            pn_scope="two-body",
        )
        one_pn = run_simulation(positions, velocities, masses, G, cfg_1pn)

        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg_2pn = SimulationConfig(
            dt=dt,
            steps=steps,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="2pn",
            c=1e8,
            pn_scope="two-body",
        )
        two_pn = run_simulation(positions, velocities, masses, G, cfg_2pn)

        self.assertLess(float(np.max(np.abs(two_pn["traj"] - one_pn["traj"]))), 1e-4)
        self.assertLess(float(np.max(np.abs(two_pn["vel"] - one_pn["vel"]))), 1e-3)

    def test_2pn_non_two_body_scope_stays_closed(self):
        positions, velocities, masses, G = mercury_relativistic_demo()
        cfg = SimulationConfig(
            dt=0.001,
            steps=5,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="mercury-relativistic",
            gravity_model="2pn",
            c=63239.7263,
            pn_scope="eih",
        )
        with self.assertRaises(NotImplementedError) as ctx:
            run_simulation(positions, velocities, masses, G, cfg)
        message = str(ctx.exception).lower()
        self.assertIn("hamiltonian", message)
        self.assertIn("2pn", message)
        self.assertIn("not implemented", message)

    def test_2pn_three_body_many_body_milestone_runs(self):
        positions, velocities, masses, G = star_earth_jupiter()
        cfg = SimulationConfig(
            dt=0.001,
            steps=2,
            save_every=1,
            eps=1e-3,
            integrator="rk4",
            preset="star-earth-jupiter",
            gravity_model="2pn",
            c=63239.7263,
            pn_scope="eih",
        )
        results = run_simulation(positions, velocities, masses, G, cfg)
        self.assertEqual(results["energy_kind"], "2pn-adm-3body")
        self.assertEqual(results["traj"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
