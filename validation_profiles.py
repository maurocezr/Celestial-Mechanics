from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolarSystemValidationProfile:
    name: str
    years: float
    dt: float
    save_every: int
    note: str


SOLAR_SYSTEM_VALIDATION_PROFILES = {
    "inner-planets": SolarSystemValidationProfile(
        name="inner-planets",
        years=10.0,
        dt=2e-4,
        save_every=5,
        note="Suitable for Mercury/Venus/Earth/Mars periapsis validation.",
    ),
    "through-jupiter": SolarSystemValidationProfile(
        name="through-jupiter",
        years=30.0,
        dt=2e-4,
        save_every=10,
        note="Long enough to capture at least two Jupiter periapsis passages for precession validation.",
    ),
    "outer-planets": SolarSystemValidationProfile(
        name="outer-planets",
        years=500.0,
        dt=5e-4,
        save_every=20,
        note="Long enough to cover Neptune and Pluto under the current two-period precession-scoring rule.",
    ),
}
