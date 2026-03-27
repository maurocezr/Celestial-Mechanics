import unittest

from validation_profiles import SOLAR_SYSTEM_VALIDATION_PROFILES


class ValidationProfileTests(unittest.TestCase):
    def test_profiles_cover_inner_and_longer_horizon_workflows(self):
        self.assertIn("inner-planets", SOLAR_SYSTEM_VALIDATION_PROFILES)
        self.assertIn("through-jupiter", SOLAR_SYSTEM_VALIDATION_PROFILES)
        self.assertIn("outer-planets", SOLAR_SYSTEM_VALIDATION_PROFILES)
        self.assertLess(
            SOLAR_SYSTEM_VALIDATION_PROFILES["inner-planets"].years,
            SOLAR_SYSTEM_VALIDATION_PROFILES["through-jupiter"].years,
        )


if __name__ == "__main__":
    unittest.main()
