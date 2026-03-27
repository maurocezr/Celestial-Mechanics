import unittest

import numpy as np

from analysis import (
    canonicalize_body_name,
    estimate_precession_rate,
    evaluate_precession_tolerance,
    planetary_orbital_period_years,
    planetary_precession_reference,
    precession_tolerance_for_body,
    summarize_precession_shift,
    summarize_precession_reference_comparison,
)


class PlanetaryPrecessionReferenceTests(unittest.TestCase):
    def test_canonicalize_horizons_style_names(self):
        self.assertEqual(canonicalize_body_name("Mercury Barycenter"), "mercury")
        self.assertEqual(canonicalize_body_name("Earth-Moon Barycenter"), "earth")
        self.assertEqual(canonicalize_body_name("Neptune"), "neptune")

    def test_mercury_reference_contains_literature_and_formula_values(self):
        reference = planetary_precession_reference("Mercury")
        self.assertIsNotNone(reference)
        self.assertAlmostEqual(reference["literature_arcsec_per_century"], 42.98, places=2)
        self.assertAlmostEqual(reference["formula_arcsec_per_century"], 42.98, delta=0.5)

    def test_pluto_reference_uses_formula_when_literature_value_is_not_bundled(self):
        reference = planetary_precession_reference("Pluto")
        self.assertIsNotNone(reference)
        self.assertNotIn("literature_arcsec_per_century", reference)
        self.assertGreater(reference["formula_arcsec_per_century"], 0.0)

        comparison = summarize_precession_reference_comparison(
            reference["formula_arcsec_per_century"],
            "Pluto",
        )
        self.assertAlmostEqual(comparison["formula_error_arcsec_per_century"], 0.0, places=12)
        self.assertNotIn("literature_error_arcsec_per_century", comparison)

    def test_precession_shift_reports_insufficient_data_when_no_periapsis_exists(self):
        times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        traj = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        results = {"times": times, "traj": traj}
        summary = summarize_precession_shift(results, results)
        self.assertEqual(summary["status"], "insufficient-data")
        self.assertIn("periapsis", summary["message"].lower())

    def test_precession_tolerance_defaults_exist_for_inner_planets(self):
        self.assertEqual(precession_tolerance_for_body("Mercury"), 5.0)
        self.assertEqual(precession_tolerance_for_body("Earth"), 1.0)
        self.assertEqual(precession_tolerance_for_body("Jupiter"), 0.01)

    def test_precession_tolerance_evaluates_pass_fail_and_insufficient_data(self):
        passing = {
            "body_name": "Mercury",
            "status": "ok",
            "literature_error_arcsec_per_century": 2.0,
        }
        failing = {
            "body_name": "Earth",
            "status": "ok",
            "literature_error_arcsec_per_century": 2.0,
        }
        insufficient = {
            "body_name": "Jupiter",
            "status": "insufficient-data",
        }

        self.assertEqual(evaluate_precession_tolerance(passing)["tolerance_status"], "pass")
        self.assertEqual(evaluate_precession_tolerance(failing)["tolerance_status"], "fail")
        self.assertEqual(evaluate_precession_tolerance(insufficient)["tolerance_status"], "insufficient-data")

    def test_planetary_orbital_period_years_matches_expected_scale(self):
        self.assertAlmostEqual(planetary_orbital_period_years("Earth"), 1.0, places=3)
        self.assertGreater(planetary_orbital_period_years("Jupiter"), 11.0)

    def test_frequency_based_precession_estimator_recovers_known_rate(self):
        period = 1.0
        eccentricity = 0.2
        semi_latus_rectum = 1.0 - eccentricity * eccentricity
        dot_omega = 0.05
        times = np.linspace(0.0, 12.0, 24001, dtype=np.float64)
        mean_anomaly = 2.0 * np.pi * times / period
        periapsis_angle = dot_omega * times
        radius = semi_latus_rectum / (1.0 + eccentricity * np.cos(mean_anomaly))
        phase = mean_anomaly + periapsis_angle
        rel = np.column_stack(
            [
                radius * np.cos(phase),
                radius * np.sin(phase),
                np.zeros_like(radius),
            ]
        )
        traj = np.zeros((times.size, 2, 3), dtype=np.float64)
        traj[:, 1, :] = rel

        measured = estimate_precession_rate(times, traj)
        self.assertAlmostEqual(measured, dot_omega, delta=1e-3)

    def test_precession_shift_rejects_runs_shorter_than_two_nominal_periods(self):
        times = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        traj = np.zeros((4, 2, 3), dtype=np.float64)
        results = {"times": times, "traj": traj}
        summary = summarize_precession_shift(results, results, body_name="Saturn")
        self.assertEqual(summary["status"], "insufficient-data")
        self.assertIn("two nominal orbital periods", summary["message"])


if __name__ == "__main__":
    unittest.main()
