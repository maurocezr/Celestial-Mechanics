from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from analysis import periapsis_samples
from nbody import DAYS_PER_YEAR, SimulationConfig, run_simulation
from relativity import solve_two_body_adm_2pn_momentum_from_velocity


DOUBLE_PULSAR_MASS_A_MSUN = 1.338185
DOUBLE_PULSAR_MASS_B_MSUN = 1.248868
DOUBLE_PULSAR_ECCENTRICITY = 0.087777023
DOUBLE_PULSAR_PERIOD_DAYS = 0.1022515592973
DOUBLE_PULSAR_PROJECTED_SEMI_MAJOR_AXIS_LIGHT_SECONDS = 1.415032
DOUBLE_PULSAR_C_AU_PER_YEAR = 63239.7263
DOUBLE_PULSAR_NAMES = np.asarray(['PSR J0737-3039A', 'PSR J0737-3039B'], dtype='U64')


def light_second_in_au(c_au_per_year: float = DOUBLE_PULSAR_C_AU_PER_YEAR) -> float:
    return float(c_au_per_year / (DAYS_PER_YEAR * 86400.0))


def projected_semi_major_axis_light_seconds(
    relative_semi_major_axis_au: float,
    masses: np.ndarray,
    sin_inclination: float,
    *,
    pulsar_index: int = 0,
    c_au_per_year: float = DOUBLE_PULSAR_C_AU_PER_YEAR,
) -> float:
    pair_masses = np.asarray(masses, dtype=np.float64)
    total_mass = float(np.sum(pair_masses))
    companion_mass = float(total_mass - pair_masses[pulsar_index])
    barycentric_semi_major_axis_au = float(relative_semi_major_axis_au) * companion_mass / total_mass
    projected_au = barycentric_semi_major_axis_au * float(sin_inclination)
    return float(projected_au / light_second_in_au(c_au_per_year))


def double_pulsar_newtonian_guess() -> dict:
    """Return the Newtonian periapsis-state guess used by the current preset."""
    G = 4.0 * math.pi * math.pi
    total_mass = DOUBLE_PULSAR_MASS_A_MSUN + DOUBLE_PULSAR_MASS_B_MSUN
    orbital_period_years = DOUBLE_PULSAR_PERIOD_DAYS / DAYS_PER_YEAR
    semi_major_axis = (total_mass * orbital_period_years * orbital_period_years) ** (1.0 / 3.0)
    periapsis = semi_major_axis * (1.0 - DOUBLE_PULSAR_ECCENTRICITY)
    relative_speed_periapsis = math.sqrt(
        G * total_mass * (1.0 + DOUBLE_PULSAR_ECCENTRICITY) / (semi_major_axis * (1.0 - DOUBLE_PULSAR_ECCENTRICITY))
    )
    sin_inclination = DOUBLE_PULSAR_PROJECTED_SEMI_MAJOR_AXIS_LIGHT_SECONDS / projected_semi_major_axis_light_seconds(
        semi_major_axis,
        np.asarray([DOUBLE_PULSAR_MASS_A_MSUN, DOUBLE_PULSAR_MASS_B_MSUN], dtype=np.float64),
        1.0,
    )
    return {
        "periapsis_au": periapsis,
        "relative_speed_au_per_year": relative_speed_periapsis,
        "sin_inclination": float(np.clip(sin_inclination, 1.0e-9, 1.0)),
        "masses": np.asarray([DOUBLE_PULSAR_MASS_A_MSUN, DOUBLE_PULSAR_MASS_B_MSUN], dtype=np.float64),
        "G": G,
    }


def build_periapsis_state(
    periapsis_au: float,
    relative_speed_au_per_year: float,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_mass = float(np.sum(masses))
    positions = np.asarray([
        [-periapsis_au * (masses[1] / total_mass), 0.0, 0.0],
        [periapsis_au * (masses[0] / total_mass), 0.0, 0.0],
    ], dtype=np.float64)
    velocities = np.asarray([
        [0.0, -relative_speed_au_per_year * (masses[1] / total_mass), 0.0],
        [0.0, relative_speed_au_per_year * (masses[0] / total_mass), 0.0],
    ], dtype=np.float64)
    return positions, velocities


def _quadratic_extremum_offset(y_prev: float, y_curr: float, y_next: float) -> float:
    denom = y_prev - 2.0 * y_curr + y_next
    if abs(denom) < 1e-18:
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -1.0, 1.0))


def _quadratic_sample(y_prev: float, y_curr: float, y_next: float, offset: float) -> float:
    a = 0.5 * (y_prev - 2.0 * y_curr + y_next)
    b = 0.5 * (y_next - y_prev)
    return float(a * offset * offset + b * offset + y_curr)


def apoapsis_samples(times: np.ndarray, traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = traj[:, 1] - traj[:, 0]
    radii = np.linalg.norm(rel, axis=1)

    indices = []
    for center in range(1, radii.shape[0] - 1):
        if radii[center] >= radii[center - 1] and radii[center] > radii[center + 1]:
            indices.append(center)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("No apoapsis passages were found in the provided trajectory")

    sample_times = np.empty(idx.size, dtype=np.float64)
    sample_radii = np.empty(idx.size, dtype=np.float64)
    for out_i, center in enumerate(idx):
        offset = _quadratic_extremum_offset(-radii[center - 1], -radii[center], -radii[center + 1])
        dt = times[center + 1] - times[center]
        sample_times[out_i] = times[center] + offset * dt
        sample_radii[out_i] = _quadratic_sample(
            radii[center - 1],
            radii[center],
            radii[center + 1],
            offset,
        )
    return sample_times, sample_radii


def summarize_realized_period_and_eccentricity(results: dict) -> dict:
    peri_times, _, peri_radii = periapsis_samples(results["times"], results["traj"])
    apo_times, apo_radii = apoapsis_samples(results["times"], results["traj"])

    if peri_times.size < 2:
        raise ValueError("At least two periapsis passages are required to estimate the realized period")

    period_days = float(np.mean(np.diff(peri_times)) * DAYS_PER_YEAR)
    mean_rp = float(np.mean(peri_radii))

    apo_between_peri = []
    for start, stop in zip(peri_times[:-1], peri_times[1:]):
        mask = (apo_times > start) & (apo_times < stop)
        if np.any(mask):
            apo_between_peri.append(float(np.mean(apo_radii[mask])))
    if not apo_between_peri:
        raise ValueError("No apoapsis passages were bracketed by periapsis passages")
    mean_ra = float(np.mean(np.asarray(apo_between_peri, dtype=np.float64)))
    eccentricity = float((mean_ra - mean_rp) / (mean_ra + mean_rp))
    return {
        "period_days": period_days,
        "eccentricity": eccentricity,
        "mean_periapsis_au": mean_rp,
        "mean_apoapsis_au": mean_ra,
        "periapsis_count": int(peri_times.size),
    }


def evaluate_double_pulsar_state(
    *,
    periapsis_au: float,
    relative_speed_au_per_year: float,
    sin_inclination: float,
    years: float,
    dt: float,
    c: float = DOUBLE_PULSAR_C_AU_PER_YEAR,
    eps: float = 0.0,
    save_every: int = 1,
    masses: np.ndarray | None = None,
    record_energy: bool = True,
) -> dict:
    guess = double_pulsar_newtonian_guess()
    pair_masses = guess["masses"] if masses is None else np.asarray(masses, dtype=np.float64)
    G = float(guess["G"])
    positions, velocities = build_periapsis_state(periapsis_au, relative_speed_au_per_year, pair_masses)
    steps = max(1, int(round(years / dt)))
    cfg = SimulationConfig(
        dt=float(dt),
        steps=steps,
        save_every=int(save_every),
        eps=float(eps),
        integrator="rk4",
        preset="psr-j0737-3039ab-fit",
        body_names=DOUBLE_PULSAR_NAMES.copy(),
        gravity_model="2pn",
        c=float(c),
        pn_scope="two-body",
    )
    results = run_simulation(positions, velocities, pair_masses, G, cfg, record_energy=record_energy)
    summary = summarize_realized_period_and_eccentricity(results)
    return {
        "results": results,
        "period_days": summary["period_days"],
        "eccentricity": summary["eccentricity"],
        "mean_semi_major_axis_relative_au": 0.5 * (summary["mean_periapsis_au"] + summary["mean_apoapsis_au"]),
        "mean_periapsis_au": summary["mean_periapsis_au"],
        "mean_apoapsis_au": summary["mean_apoapsis_au"],
        "projected_semi_major_axis_light_seconds": projected_semi_major_axis_light_seconds(
            0.5 * (summary["mean_periapsis_au"] + summary["mean_apoapsis_au"]),
            pair_masses,
            sin_inclination,
            c_au_per_year=c,
        ),
        "periapsis_count": summary["periapsis_count"],
        "periapsis_au": float(periapsis_au),
        "relative_speed_au_per_year": float(relative_speed_au_per_year),
        "sin_inclination": float(sin_inclination),
        "masses": pair_masses,
        "G": G,
    }


def build_double_pulsar_2pn_initial_state(
    *,
    c: float = DOUBLE_PULSAR_C_AU_PER_YEAR,
) -> dict:
    guess = double_pulsar_newtonian_guess()
    pair_masses = np.asarray(guess["masses"], dtype=np.float64)
    positions, velocities = build_periapsis_state(
        float(guess["periapsis_au"]),
        float(guess["relative_speed_au_per_year"]),
        pair_masses,
    )
    relative_position = positions[0] - positions[1]
    relative_velocity = velocities[0] - velocities[1]
    relative_momentum = solve_two_body_adm_2pn_momentum_from_velocity(
        relative_position,
        relative_velocity,
        pair_masses,
        float(guess["G"]),
        c,
    )
    total_mass = float(np.sum(pair_masses))
    canonical_momenta = np.asarray([relative_momentum, -relative_momentum], dtype=np.float64)
    return {
        "positions": positions,
        "velocities": velocities,
        "relative_position": relative_position,
        "relative_velocity": relative_velocity,
        "relative_momentum": relative_momentum,
        "canonical_momenta": canonical_momenta,
        "periapsis_au": float(guess["periapsis_au"]),
        "relative_speed_au_per_year": float(guess["relative_speed_au_per_year"]),
        "sin_inclination": float(guess["sin_inclination"]),
        "semi_major_axis_au": float(guess["periapsis_au"] / (1.0 - DOUBLE_PULSAR_ECCENTRICITY)),
        "projected_semi_major_axis_light_seconds": projected_semi_major_axis_light_seconds(
            float(guess["periapsis_au"] / (1.0 - DOUBLE_PULSAR_ECCENTRICITY)),
            pair_masses,
            float(guess["sin_inclination"]),
            c_au_per_year=c,
        ),
        "period_days": DOUBLE_PULSAR_PERIOD_DAYS,
        "eccentricity": DOUBLE_PULSAR_ECCENTRICITY,
        "masses": pair_masses,
        "G": float(guess["G"]),
    }


def fit_double_pulsar_2pn(
    *,
    c: float = DOUBLE_PULSAR_C_AU_PER_YEAR,
    **_: object,
) -> dict:
    """Return the direct canonical 2PN initial state from the Newtonian periapsis guess.

    The previous iterative fitter has been retired. This function is kept as a
    compatibility wrapper for callers and now returns the canonical initial
    conditions obtained from the Newtonian `(r0, v0)` construction followed by
    the Hamiltonian inversion `v = dH/dp`.
    """
    return build_double_pulsar_2pn_initial_state(c=c)


def save_fitted_state(path: str | Path, fit: dict) -> None:
    output_path = Path(path)
    np.savez(
        output_path,
        positions=fit["positions"],
        velocities=fit["velocities"],
        canonical_momenta=fit["canonical_momenta"],
        masses=fit["masses"],
        G=np.asarray(fit["G"]),
        names=DOUBLE_PULSAR_NAMES,
        note=np.asarray(
            "Units: AU, yr, Msun; G=4*pi^2; direct Double Pulsar 2PN canonical initial state from Newtonian periapsis guess"
        ),
        sin_inclination=np.asarray(fit["sin_inclination"]),
        projected_semi_major_axis_light_seconds=np.asarray(fit["projected_semi_major_axis_light_seconds"]),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a PSR J0737-3039A/B direct 2PN canonical initial state from the Newtonian periapsis guess and the ADM velocity-momentum inversion."
    )
    parser.add_argument("--c", type=float, default=DOUBLE_PULSAR_C_AU_PER_YEAR, help="Effective speed of light in AU/yr")
    parser.add_argument("--save-state", type=str, default=None, help="Optional NPZ path for the fitted initial state")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fit = fit_double_pulsar_2pn(
        c=args.c,
    )
    print("direct Double Pulsar 2PN canonical initial state")
    print(f"  period_days={fit['period_days']:.13f}")
    print(f"  eccentricity={fit['eccentricity']:.9f}")
    print(f"  projected_semi_major_axis_light_seconds={fit['projected_semi_major_axis_light_seconds']:.12f}")
    print(f"  periapsis_au={fit['periapsis_au']:.15e}")
    print(f"  semi_major_axis_au={fit['semi_major_axis_au']:.15e}")
    print(f"  relative_speed_au_per_year={fit['relative_speed_au_per_year']:.15e}")
    print(f"  sin_inclination={fit['sin_inclination']:.12f}")
    print(f"  relative_position_au={fit['relative_position']}")
    print(f"  relative_velocity_au_per_year={fit['relative_velocity']}")
    print(f"  relative_momentum_canonical={fit['relative_momentum']}")
    if args.save_state:
        save_fitted_state(args.save_state, fit)
        print(f"saved_state={args.save_state}")


if __name__ == "__main__":
    main()
