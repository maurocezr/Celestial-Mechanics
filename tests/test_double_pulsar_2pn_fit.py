import unittest

from fit_double_pulsar_2pn import (
    DOUBLE_PULSAR_ECCENTRICITY,
    DOUBLE_PULSAR_PERIOD_DAYS,
    DOUBLE_PULSAR_PROJECTED_SEMI_MAJOR_AXIS_LIGHT_SECONDS,
    build_double_pulsar_2pn_initial_state,
    double_pulsar_newtonian_guess,
    evaluate_double_pulsar_state,
    fit_double_pulsar_2pn,
)
from relativity import two_body_adm_2pn_rhs


class DoublePulsar2PNFitTests(unittest.TestCase):
    def test_direct_builder_matches_newtonian_target_observables(self):
        guess = double_pulsar_newtonian_guess()
        direct = build_double_pulsar_2pn_initial_state()
        self.assertAlmostEqual(direct["period_days"], DOUBLE_PULSAR_PERIOD_DAYS, places=12)
        self.assertAlmostEqual(direct["eccentricity"], DOUBLE_PULSAR_ECCENTRICITY, places=12)
        self.assertLess(
            abs(direct["projected_semi_major_axis_light_seconds"] - DOUBLE_PULSAR_PROJECTED_SEMI_MAJOR_AXIS_LIGHT_SECONDS),
            1e-8,
        )
        self.assertAlmostEqual(direct["periapsis_au"], guess["periapsis_au"], places=15)
        self.assertAlmostEqual(direct["relative_speed_au_per_year"], guess["relative_speed_au_per_year"], places=12)

    def test_direct_builder_momentum_reproduces_requested_velocity(self):
        direct = fit_double_pulsar_2pn()
        solved_velocity, _ = two_body_adm_2pn_rhs(
            direct["relative_position"],
            direct["relative_momentum"],
            direct["masses"],
            direct["G"],
            63239.7263,
        )
        self.assertLess(float(abs(solved_velocity[0] - direct["relative_velocity"][0])), 1e-11)
        self.assertLess(float(abs(solved_velocity[1] - direct["relative_velocity"][1])), 1e-11)
        self.assertLess(float(abs(solved_velocity[2] - direct["relative_velocity"][2])), 1e-11)

    def test_evaluate_state_still_runs_from_direct_guess(self):
        guess = double_pulsar_newtonian_guess()
        summary = evaluate_double_pulsar_state(
            periapsis_au=guess["periapsis_au"],
            relative_speed_au_per_year=guess["relative_speed_au_per_year"],
            sin_inclination=guess["sin_inclination"],
            years=0.0015,
            dt=2e-6,
            save_every=1,
            record_energy=False,
        )
        self.assertGreater(summary["periapsis_count"], 1)


if __name__ == "__main__":
    unittest.main()
