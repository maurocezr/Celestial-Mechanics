from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from relativity import total_energy_1pn


Array = np.ndarray
CANONICAL_PLANETARY_PRECESSION_ARCSEC_PER_CENTURY = {
    "mercury": 42.98,
    "venus": 8.62,
    "earth": 3.84,
    "mars": 1.35,
    "jupiter": 0.062,
    "saturn": 0.014,
    "uranus": 0.0024,
    "neptune": 0.0008,
}
# Mean orbital elements for formula-based reference values in AU and Julian years.
PLANETARY_REFERENCE_ELEMENTS = {
    "mercury": {"semi_major_axis_au": 0.387098, "eccentricity": 0.205630},
    "venus": {"semi_major_axis_au": 0.723332, "eccentricity": 0.006772},
    "earth": {"semi_major_axis_au": 1.000000, "eccentricity": 0.0167086},
    "mars": {"semi_major_axis_au": 1.523679, "eccentricity": 0.0934},
    "jupiter": {"semi_major_axis_au": 5.2044, "eccentricity": 0.0489},
    "saturn": {"semi_major_axis_au": 9.5826, "eccentricity": 0.0565},
    "uranus": {"semi_major_axis_au": 19.2184, "eccentricity": 0.0463},
    "neptune": {"semi_major_axis_au": 30.1104, "eccentricity": 0.009456},
    "pluto": {"semi_major_axis_au": 39.482, "eccentricity": 0.2488},
}
PLANETARY_NAME_ALIASES = {
    "sun": "sun",
    "mercury": "mercury",
    "mercurybarycenter": "mercury",
    "venus": "venus",
    "venusbarycenter": "venus",
    "earth": "earth",
    "earthmoonbarycenter": "earth",
    "mars": "mars",
    "marsbarycenter": "mars",
    "jupiter": "jupiter",
    "jupiterbarycenter": "jupiter",
    "saturn": "saturn",
    "saturnbarycenter": "saturn",
    "uranus": "uranus",
    "uranusbarycenter": "uranus",
    "neptune": "neptune",
    "neptunebarycenter": "neptune",
    "pluto": "pluto",
    "plutobarycenter": "pluto",
}
DEFAULT_PRECESSION_TOLERANCES_ARCSEC_PER_CENTURY = {
    "mercury": 5.0,
    "venus": 2.0,
    "earth": 1.0,
    "mars": 0.5,
    "jupiter": 0.01,
}


def load_results(path: str | Path) -> dict:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_scalar(results: dict, key: str, default=None):
    value = results.get(key, default)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def canonicalize_body_name(name: str) -> str:
    normalized = ''.join(ch for ch in str(name).strip().lower() if ch.isalnum())
    return PLANETARY_NAME_ALIASES.get(normalized, normalized)


def relative_trajectory(traj: Array, primary_index: int = 0, secondary_index: int = 1) -> Array:
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError(f"traj must have shape (T, N, 3); got {traj.shape}")
    if not (0 <= primary_index < traj.shape[1] and 0 <= secondary_index < traj.shape[1]):
        raise ValueError("primary_index and secondary_index must index valid bodies")
    if primary_index == secondary_index:
        raise ValueError("primary_index and secondary_index must be different")
    return traj[:, secondary_index, :] - traj[:, primary_index, :]


def relative_velocity(vel: Array, primary_index: int = 0, secondary_index: int = 1) -> Array:
    vel = np.asarray(vel, dtype=np.float64)
    if vel.ndim != 3 or vel.shape[2] != 3:
        raise ValueError(f"vel must have shape (T, N, 3); got {vel.shape}")
    if not (0 <= primary_index < vel.shape[1] and 0 <= secondary_index < vel.shape[1]):
        raise ValueError("primary_index and secondary_index must index valid bodies")
    if primary_index == secondary_index:
        raise ValueError("primary_index and secondary_index must be different")
    return vel[:, secondary_index, :] - vel[:, primary_index, :]


def estimate_periapsis_indices(radii: Array) -> Array:
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 3:
        raise ValueError("radii must be a 1D array with at least 3 samples")
    mask = (radii[1:-1] < radii[:-2]) & (radii[1:-1] < radii[2:])
    return np.nonzero(mask)[0] + 1


def _quadratic_minimum_offset(y_prev: float, y_curr: float, y_next: float) -> float:
    denom = y_prev - 2.0 * y_curr + y_next
    if denom == 0.0:
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -1.0, 1.0))


def _quadratic_sample(y_prev: float, y_curr: float, y_next: float, offset: float) -> float:
    a = 0.5 * (y_prev - 2.0 * y_curr + y_next)
    b = 0.5 * (y_next - y_prev)
    return float(a * offset * offset + b * offset + y_curr)


def periapsis_samples(
    times: Array,
    traj: Array,
    primary_index: int = 0,
    secondary_index: int = 1,
) -> tuple[Array, Array, Array]:
    times = np.asarray(times, dtype=np.float64)
    rel = relative_trajectory(traj, primary_index=primary_index, secondary_index=secondary_index)
    radii = np.linalg.norm(rel, axis=1)
    idx, offsets = estimate_periapsis_event_offsets(radii)
    sample_times, sample_positions, sample_radii = interpolate_periapsis_events(times, rel, idx, offsets)
    return sample_times, sample_positions, sample_radii


def periapsis_angles(traj: Array, primary_index: int = 0, secondary_index: int = 1) -> Array:
    rel = relative_trajectory(traj, primary_index=primary_index, secondary_index=secondary_index)
    radii = np.linalg.norm(rel, axis=1)
    idx = estimate_periapsis_indices(radii)
    if idx.size == 0:
        raise ValueError("No periapsis passages were found in the provided trajectory")
    return np.unwrap(np.arctan2(rel[idx, 1], rel[idx, 0]))


def estimate_periapsis_event_offsets(radii: Array) -> tuple[Array, Array]:
    idx = estimate_periapsis_indices(radii)
    if idx.size == 0:
        raise ValueError("No periapsis passages were found in the provided trajectory")
    offsets = np.empty(idx.size, dtype=np.float64)
    for out_i, center in enumerate(idx):
        offsets[out_i] = _quadratic_minimum_offset(radii[center - 1], radii[center], radii[center + 1])
    return idx, offsets


def interpolate_periapsis_events(
    times: Array,
    rel: Array,
    indices: Array,
    offsets: Array,
) -> tuple[Array, Array, Array]:
    sample_times = np.empty(indices.size, dtype=np.float64)
    sample_positions = np.empty((indices.size, 3), dtype=np.float64)
    sample_radii = np.empty(indices.size, dtype=np.float64)

    for out_i, center in enumerate(indices):
        offset = float(offsets[out_i])
        dt = times[center + 1] - times[center]
        sample_times[out_i] = times[center] + offset * dt
        for axis in range(3):
            sample_positions[out_i, axis] = _quadratic_sample(
                rel[center - 1, axis],
                rel[center, axis],
                rel[center + 1, axis],
                offset,
            )
        sample_radii[out_i] = np.linalg.norm(sample_positions[out_i])

    return sample_times, sample_positions, sample_radii


def estimate_precession_rate_from_frequencies(
    times: Array,
    traj: Array,
    primary_index: int = 0,
    secondary_index: int = 1,
) -> dict[str, Array | float]:
    times = np.asarray(times, dtype=np.float64)
    rel = relative_trajectory(traj, primary_index=primary_index, secondary_index=secondary_index)
    radii = np.linalg.norm(rel, axis=1)
    peri_idx, peri_offsets = estimate_periapsis_event_offsets(radii)
    peri_times, peri_positions, _ = interpolate_periapsis_events(times, rel, peri_idx, peri_offsets)
    if peri_times.size < 2:
        raise ValueError("At least two periapsis passages are required to estimate a precession rate")

    phase = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    peri_phase = np.empty(peri_idx.size, dtype=np.float64)
    for out_i, center in enumerate(peri_idx):
        peri_phase[out_i] = _quadratic_sample(
            phase[center - 1],
            phase[center],
            phase[center + 1],
            float(peri_offsets[out_i]),
        )

    peri_angles = np.unwrap(np.arctan2(peri_positions[:, 1], peri_positions[:, 0]))
    radial_periods = np.diff(peri_times)
    azimuthal_advances = np.diff(peri_phase)
    radial_frequencies = (2.0 * math.pi) / radial_periods
    azimuthal_frequencies = azimuthal_advances / radial_periods
    cycle_precession_rates = azimuthal_frequencies - radial_frequencies

    return {
        "peri_times": peri_times,
        "peri_angles": peri_angles,
        "peri_phase": peri_phase,
        "radial_periods": radial_periods,
        "azimuthal_advances": azimuthal_advances,
        "radial_frequencies": radial_frequencies,
        "azimuthal_frequencies": azimuthal_frequencies,
        "cycle_precession_rates": cycle_precession_rates,
        "mean_precession_rate": float(np.mean(cycle_precession_rates)),
    }


def estimate_precession_rate(
    times: Array,
    traj: Array,
    primary_index: int = 0,
    secondary_index: int = 1,
) -> float:
    components = estimate_precession_rate_from_frequencies(
        times, traj, primary_index=primary_index, secondary_index=secondary_index
    )
    return float(components["mean_precession_rate"])


def arcsec_per_century_from_rad_per_year(rate_rad_per_year: float) -> float:
    return float(rate_rad_per_year * 206264.80624709636 * 100.0)


def planetary_precession_formula_arcsec_per_century(
    body_name: str,
    *,
    primary_mass_msun: float = 1.0,
    secondary_mass_msun: float = 0.0,
    c_au_per_year: float = 63239.7263,
    G: float = 4.0 * math.pi * math.pi,
) -> float:
    canonical_name = canonicalize_body_name(body_name)
    if canonical_name not in PLANETARY_REFERENCE_ELEMENTS:
        raise KeyError(f"No orbital elements are available for body {body_name!r}")
    elements = PLANETARY_REFERENCE_ELEMENTS[canonical_name]
    semi_major_axis = float(elements["semi_major_axis_au"])
    eccentricity = float(elements["eccentricity"])
    total_mass = primary_mass_msun + secondary_mass_msun
    period_years = math.sqrt(semi_major_axis**3 / total_mass)
    advance_per_orbit = (6.0 * math.pi * G * total_mass) / (
        c_au_per_year * c_au_per_year * semi_major_axis * (1.0 - eccentricity * eccentricity)
    )
    return arcsec_per_century_from_rad_per_year(advance_per_orbit / period_years)


def planetary_orbital_period_years(
    body_name: str,
    *,
    primary_mass_msun: float = 1.0,
    secondary_mass_msun: float = 0.0,
) -> float:
    canonical_name = canonicalize_body_name(body_name)
    if canonical_name not in PLANETARY_REFERENCE_ELEMENTS:
        raise KeyError(f"No orbital elements are available for body {body_name!r}")
    semi_major_axis = float(PLANETARY_REFERENCE_ELEMENTS[canonical_name]["semi_major_axis_au"])
    total_mass = primary_mass_msun + secondary_mass_msun
    return math.sqrt(semi_major_axis**3 / total_mass)


def planetary_precession_reference(body_name: str) -> dict | None:
    canonical_name = canonicalize_body_name(body_name)
    if canonical_name not in PLANETARY_REFERENCE_ELEMENTS and canonical_name not in CANONICAL_PLANETARY_PRECESSION_ARCSEC_PER_CENTURY:
        return None

    literature_value = CANONICAL_PLANETARY_PRECESSION_ARCSEC_PER_CENTURY.get(canonical_name)
    formula_value = planetary_precession_formula_arcsec_per_century(canonical_name)
    reference = {
        "body_name": canonical_name.title(),
        "formula_arcsec_per_century": formula_value,
        "formula_note": "Schwarzschild 1PN perihelion-advance estimate from bundled mean orbital elements.",
    }
    if literature_value is not None:
        reference["literature_arcsec_per_century"] = literature_value
        reference["literature_note"] = "Canonical Solar System GR perihelion advances commonly quoted in the literature."
    return reference


def summarize_precession_reference_comparison(
    measured_arcsec_per_century: float,
    body_name: str,
) -> dict:
    reference = planetary_precession_reference(body_name)
    comparison = {
        "body_name": canonicalize_body_name(body_name).title(),
        "measured_arcsec_per_century": float(measured_arcsec_per_century),
    }
    if reference is None:
        return comparison

    comparison.update(reference)
    comparison["formula_error_arcsec_per_century"] = (
        comparison["measured_arcsec_per_century"] - comparison["formula_arcsec_per_century"]
    )
    if "literature_arcsec_per_century" in comparison:
        literature_value = comparison["literature_arcsec_per_century"]
        comparison["literature_error_arcsec_per_century"] = comparison["measured_arcsec_per_century"] - literature_value
        comparison["literature_relative_error_percent"] = (
            100.0 * comparison["literature_error_arcsec_per_century"] / literature_value
        )
    return comparison


def precession_tolerance_for_body(body_name: str) -> float | None:
    return DEFAULT_PRECESSION_TOLERANCES_ARCSEC_PER_CENTURY.get(canonicalize_body_name(body_name))


def two_body_orbital_elements(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
) -> dict:
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions and velocities must both have shape (T, 3)")
    if masses.shape != (2,):
        raise ValueError("two_body_orbital_elements requires exactly two masses")

    total_mass = float(masses[0] + masses[1])
    mu = G * total_mass
    radius = np.linalg.norm(positions, axis=1)
    speed2 = np.sum(velocities * velocities, axis=1)

    angular_momentum = np.cross(positions, velocities)
    e_vec = np.cross(velocities, angular_momentum) / mu - positions / radius[:, None]
    eccentricity = np.linalg.norm(e_vec, axis=1)

    inv_a = 2.0 / radius - speed2 / mu
    semi_major_axis = np.where(np.abs(inv_a) > 1e-16, 1.0 / inv_a, np.inf)
    periapsis_angle = np.unwrap(np.arctan2(e_vec[:, 1], e_vec[:, 0]))

    return {
        "radius": radius,
        "speed": np.sqrt(speed2),
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "periapsis_angle": periapsis_angle,
        "specific_angular_momentum": np.linalg.norm(angular_momentum, axis=1),
        "eccentricity_vector": e_vec,
    }


def orbital_elements_series(
    times: Array,
    traj: Array,
    vel: Array,
    masses: Array,
    G: float,
    primary_index: int = 0,
    secondary_index: int = 1,
) -> dict:
    del times
    rel_pos = relative_trajectory(traj, primary_index=primary_index, secondary_index=secondary_index)
    rel_vel = relative_velocity(vel, primary_index=primary_index, secondary_index=secondary_index)
    pair_masses = np.asarray([masses[primary_index], masses[secondary_index]], dtype=np.float64)
    return two_body_orbital_elements(rel_pos, rel_vel, pair_masses, G)


def summarize_orbital_elements(
    times: Array,
    traj: Array,
    vel: Array,
    masses: Array,
    G: float,
    primary_index: int = 0,
    secondary_index: int = 1,
) -> dict:
    elements = orbital_elements_series(
        times,
        traj,
        vel,
        masses,
        G,
        primary_index=primary_index,
        secondary_index=secondary_index,
    )
    peri_rate = estimate_precession_rate(times, traj, primary_index=primary_index, secondary_index=secondary_index)
    return {
        "semi_major_axis_mean": float(np.mean(elements["semi_major_axis"])),
        "semi_major_axis_std": float(np.std(elements["semi_major_axis"])),
        "eccentricity_mean": float(np.mean(elements["eccentricity"])),
        "eccentricity_std": float(np.std(elements["eccentricity"])),
        "periapsis_angle_span": float(elements["periapsis_angle"][-1] - elements["periapsis_angle"][0]),
        "precession_rate_rad_per_year": peri_rate,
        "precession_rate_arcsec_per_century": arcsec_per_century_from_rad_per_year(peri_rate),
    }


def summarize_system_relative_to_primary(
    times: Array,
    traj: Array,
    vel: Array,
    masses: Array,
    G: float,
    primary_index: int = 0,
) -> dict[int, dict]:
    summaries: dict[int, dict] = {}
    for secondary_index in range(traj.shape[1]):
        if secondary_index == primary_index:
            continue
        summaries[secondary_index] = summarize_orbital_elements(
            times,
            traj,
            vel,
            masses,
            G,
            primary_index=primary_index,
            secondary_index=secondary_index,
        )
    return summaries


def summarize_orbital_difference(
    reference_results: dict,
    candidate_results: dict,
    *,
    primary_index: int = 0,
    secondary_index: int = 1,
) -> dict:
    G_ref = float(_load_scalar(reference_results, "G"))
    G_cand = float(_load_scalar(candidate_results, "G"))
    try:
        reference = summarize_orbital_elements(
            reference_results["times"],
            reference_results["traj"],
            reference_results["vel"],
            reference_results["masses"],
            G_ref,
            primary_index=primary_index,
            secondary_index=secondary_index,
        )
        candidate = summarize_orbital_elements(
            candidate_results["times"],
            candidate_results["traj"],
            candidate_results["vel"],
            candidate_results["masses"],
            G_cand,
            primary_index=primary_index,
            secondary_index=secondary_index,
        )
    except ValueError as exc:
        return {
            "orbital_status": "insufficient-data",
            "orbital_message": str(exc),
            "delta_semi_major_axis_mean": float("nan"),
            "delta_eccentricity_mean": float("nan"),
            "delta_precession_rate_arcsec_per_century": float("nan"),
        }
    return {
        "orbital_status": "ok",
        "reference": reference,
        "candidate": candidate,
        "delta_semi_major_axis_mean": candidate["semi_major_axis_mean"] - reference["semi_major_axis_mean"],
        "delta_eccentricity_mean": candidate["eccentricity_mean"] - reference["eccentricity_mean"],
        "delta_precession_rate_arcsec_per_century": (
            candidate["precession_rate_arcsec_per_century"] - reference["precession_rate_arcsec_per_century"]
        ),
    }


def summarize_system_orbital_differences(
    reference_results: dict,
    candidate_results: dict,
    *,
    primary_index: int = 0,
) -> dict[int, dict]:
    body_names = candidate_results.get("body_names")
    if body_names is None:
        body_names = np.asarray([str(i) for i in range(candidate_results["traj"].shape[1])], dtype="U64")

    summaries: dict[int, dict] = {}
    for secondary_index in range(candidate_results["traj"].shape[1]):
        if secondary_index == primary_index:
            continue
        summary = summarize_orbital_difference(
            reference_results,
            candidate_results,
            primary_index=primary_index,
            secondary_index=secondary_index,
        )
        summary["body_name"] = canonicalize_body_name(str(body_names[secondary_index])).title()
        summaries[secondary_index] = summary
    return summaries


def summarize_precession_shift(
    reference_results: dict,
    candidate_results: dict,
    *,
    primary_index: int = 0,
    secondary_index: int = 1,
    body_name: str | None = None,
) -> dict:
    if body_name is not None:
        try:
            min_required_years = 2.0 * planetary_orbital_period_years(body_name)
            observed_years = min(
                float(np.asarray(reference_results["times"], dtype=np.float64)[-1]),
                float(np.asarray(candidate_results["times"], dtype=np.float64)[-1]),
            )
            if observed_years < min_required_years:
                return {
                    "status": "insufficient-data",
                    "message": (
                        f"Need at least {min_required_years:.2f} years to cover two nominal orbital periods for "
                        f"{canonicalize_body_name(body_name).title()}; only {observed_years:.2f} years were provided"
                    ),
                }
        except KeyError:
            pass
    try:
        reference_rate = estimate_precession_rate(
            reference_results["times"],
            reference_results["traj"],
            primary_index=primary_index,
            secondary_index=secondary_index,
        )
        candidate_rate = estimate_precession_rate(
            candidate_results["times"],
            candidate_results["traj"],
            primary_index=primary_index,
            secondary_index=secondary_index,
        )
    except ValueError as exc:
        return {
            "status": "insufficient-data",
            "message": str(exc),
        }
    baseline_subtracted_rate = candidate_rate - reference_rate
    return {
        "status": "ok",
        "reference_rate_rad_per_year": float(reference_rate),
        "candidate_rate_rad_per_year": float(candidate_rate),
        "baseline_subtracted_rate_rad_per_year": float(baseline_subtracted_rate),
        "reference_rate_arcsec_per_century": arcsec_per_century_from_rad_per_year(reference_rate),
        "candidate_rate_arcsec_per_century": arcsec_per_century_from_rad_per_year(candidate_rate),
        "baseline_subtracted_rate_arcsec_per_century": arcsec_per_century_from_rad_per_year(baseline_subtracted_rate),
    }


def summarize_system_precession_vs_references(
    reference_results: dict,
    candidate_results: dict,
    *,
    primary_index: int = 0,
) -> dict[int, dict]:
    body_names = candidate_results.get("body_names")
    if body_names is None:
        body_names = np.asarray([str(i) for i in range(candidate_results["traj"].shape[1])], dtype="U64")

    summaries: dict[int, dict] = {}
    for secondary_index in range(candidate_results["traj"].shape[1]):
        if secondary_index == primary_index:
            continue
        shift_summary = summarize_precession_shift(
            reference_results,
            candidate_results,
            primary_index=primary_index,
            secondary_index=secondary_index,
            body_name=str(body_names[secondary_index]),
        )
        shift_summary["body_name"] = canonicalize_body_name(str(body_names[secondary_index])).title()
        if shift_summary["status"] == "ok":
            comparison = summarize_precession_reference_comparison(
                shift_summary["baseline_subtracted_rate_arcsec_per_century"],
                str(body_names[secondary_index]),
            )
            shift_summary.update(comparison)
        summaries[secondary_index] = shift_summary
    return summaries


def evaluate_precession_tolerance(summary: dict, *, absolute_tolerance_arcsec_per_century: float | None = None) -> dict:
    status = summary.get("status", "ok")
    if status != "ok":
        return {
            "tolerance_status": "insufficient-data",
            "tolerance_arcsec_per_century": absolute_tolerance_arcsec_per_century,
        }

    body_name = str(summary.get("body_name", ""))
    tolerance = absolute_tolerance_arcsec_per_century
    if tolerance is None:
        tolerance = precession_tolerance_for_body(body_name)
    result = {
        "tolerance_arcsec_per_century": tolerance,
    }
    if tolerance is None or "literature_error_arcsec_per_century" not in summary:
        result["tolerance_status"] = "unscored"
        return result

    error = abs(float(summary["literature_error_arcsec_per_century"]))
    result["tolerance_status"] = "pass" if error <= tolerance else "fail"
    return result


def summarize_system_validation(
    reference_results: dict,
    candidate_results: dict,
    *,
    primary_index: int = 0,
) -> dict[int, dict]:
    orbital = summarize_system_orbital_differences(
        reference_results,
        candidate_results,
        primary_index=primary_index,
    )
    precession = summarize_system_precession_vs_references(
        reference_results,
        candidate_results,
        primary_index=primary_index,
    )
    summaries: dict[int, dict] = {}
    for secondary_index, orbital_summary in orbital.items():
        merged = dict(orbital_summary)
        merged.update(precession.get(secondary_index, {}))
        merged.update(evaluate_precession_tolerance(merged))
        summaries[secondary_index] = merged
    return summaries


def energy_diagnostic_series(results: dict) -> dict:
    energy = np.asarray(results["energy"], dtype=np.float64)
    kind = str(_load_scalar(results, "energy_kind", "newtonian"))
    rel = (energy - energy[0]) / (abs(energy[0]) + 1e-16)
    return {
        "energy_kind": kind,
        "energy": energy,
        "relative_drift": rel,
        "max_abs_relative_drift": float(np.max(np.abs(rel))),
        "final_relative_drift": float(rel[-1]),
    }


def pn_energy_series(results: dict) -> dict:
    gravity_model = str(_load_scalar(results, "gravity_model", "newtonian"))
    if gravity_model != "1pn":
        raise ValueError("pn_energy_series is only defined for saved 1PN runs")

    c = float(_load_scalar(results, "c"))
    G = float(_load_scalar(results, "G"))
    masses = np.asarray(results["masses"], dtype=np.float64)
    traj = np.asarray(results["traj"], dtype=np.float64)
    vel = np.asarray(results["vel"], dtype=np.float64)
    pn_scope = str(_load_scalar(results, "pn_scope", "two-body"))
    energy_kind = str(_load_scalar(results, "energy_kind", "1pn"))

    energy = np.empty(traj.shape[0], dtype=np.float64)
    for i in range(traj.shape[0]):
        energy[i] = total_energy_1pn(traj[i], vel[i], masses, G, c, scope=pn_scope)
    rel = (energy - energy[0]) / (abs(energy[0]) + 1e-16)
    return {
        "energy_kind": energy_kind,
        "energy": energy,
        "relative_drift": rel,
        "max_abs_relative_drift": float(np.max(np.abs(rel))),
        "final_relative_drift": float(rel[-1]),
    }
