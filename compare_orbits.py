from __future__ import annotations

import argparse
from pathlib import Path

from analysis import energy_diagnostic_series, load_results, summarize_orbital_elements


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare orbital elements and energy diagnostics between two saved .npz runs."
    )
    parser.add_argument("reference", type=str, help="Reference .npz file, typically Newtonian")
    parser.add_argument("candidate", type=str, help="Candidate .npz file, typically 1PN")
    parser.add_argument("--primary-index", type=int, default=0, help="Central/reference body index")
    parser.add_argument("--secondary-index", type=int, default=1, help="Orbiting body index")
    return parser.parse_args(argv)


def summarize(path: str, *, primary_index: int, secondary_index: int) -> dict:
    results = load_results(path)
    summary = summarize_orbital_elements(
        results["times"],
        results["traj"],
        results["vel"],
        results["masses"],
        float(results["G"].item() if getattr(results["G"], "shape", None) == () else results["G"]),
        primary_index=primary_index,
        secondary_index=secondary_index,
    )
    energy = energy_diagnostic_series(results)
    summary["energy_kind"] = energy["energy_kind"]
    summary["energy_max_abs_relative_drift"] = energy["max_abs_relative_drift"]
    summary["energy_final_relative_drift"] = energy["final_relative_drift"]
    return summary


def print_summary(label: str, summary: dict) -> None:
    print(label)
    print(f"  semi_major_axis_mean: {summary['semi_major_axis_mean']:.12g}")
    print(f"  semi_major_axis_std: {summary['semi_major_axis_std']:.12g}")
    print(f"  eccentricity_mean: {summary['eccentricity_mean']:.12g}")
    print(f"  eccentricity_std: {summary['eccentricity_std']:.12g}")
    print(f"  periapsis_angle_span_rad: {summary['periapsis_angle_span']:.12e}")
    print(f"  precession_rate_rad_per_year: {summary['precession_rate_rad_per_year']:.12e}")
    print(f"  precession_rate_arcsec_per_century: {summary['precession_rate_arcsec_per_century']:.12f}")
    print(f"  energy_kind: {summary['energy_kind']}")
    print(f"  energy_max_abs_relative_drift: {summary['energy_max_abs_relative_drift']:.12e}")
    print(f"  energy_final_relative_drift: {summary['energy_final_relative_drift']:.12e}")


def main(argv=None):
    args = parse_args(argv)
    reference = summarize(args.reference, primary_index=args.primary_index, secondary_index=args.secondary_index)
    candidate = summarize(args.candidate, primary_index=args.primary_index, secondary_index=args.secondary_index)

    print_summary(Path(args.reference).name, reference)
    print_summary(Path(args.candidate).name, candidate)

    print("comparison")
    print(f"  delta_semi_major_axis_mean: {candidate['semi_major_axis_mean'] - reference['semi_major_axis_mean']:.12e}")
    print(f"  delta_eccentricity_mean: {candidate['eccentricity_mean'] - reference['eccentricity_mean']:.12e}")
    print(
        "  delta_precession_rate_arcsec_per_century: "
        f"{candidate['precession_rate_arcsec_per_century'] - reference['precession_rate_arcsec_per_century']:.12f}"
    )
    print(
        "  delta_energy_max_abs_relative_drift: "
        f"{candidate['energy_max_abs_relative_drift'] - reference['energy_max_abs_relative_drift']:.12e}"
    )


if __name__ == "__main__":
    main()
