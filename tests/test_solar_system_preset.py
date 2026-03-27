import unittest
from unittest.mock import patch

import numpy as np

from nbody import SOLAR_SYSTEM_BODY_NAMES, SOLAR_SYSTEM_HORIZONS_TARGETS, load_preset


class SolarSystemHorizonsPresetTests(unittest.TestCase):
    @patch("nbody.fetch_horizons_ephemeris")
    def test_solar_system_horizons_preset_uses_fixed_targets_and_names(self, fetch_mock):
        fetch_mock.return_value = (
            np.zeros((10, 3), dtype=np.float64),
            np.zeros((10, 3), dtype=np.float64),
            np.ones(10, dtype=np.float64),
            4.0 * np.pi * np.pi,
            "Units: AU, yr, Msun",
            np.asarray(["ignored"] * 10, dtype="U64"),
            "2026-01-01 00:00",
        )

        _, _, _, _, unit_note, body_names, epoch = load_preset(
            "solar-system-horizons",
            n=100,
            seed=42,
            mass_spread=0.0,
            horizons_epoch="2026-01-01 00:00",
        )

        fetch_mock.assert_called_once()
        self.assertEqual(fetch_mock.call_args.kwargs["commands"], SOLAR_SYSTEM_HORIZONS_TARGETS)
        np.testing.assert_array_equal(body_names, SOLAR_SYSTEM_BODY_NAMES)
        self.assertIn("preset=solar-system-horizons", unit_note)
        self.assertEqual(epoch, "2026-01-01 00:00")


if __name__ == "__main__":
    unittest.main()
