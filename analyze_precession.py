from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis import (
    arcsec_per_century_from_rad_per_year,
    estimate_precession_rate,
    load_results,
    periapsis_samples,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimate periapsis precession from a saved .npz trajectory.")
    parser.add_argument("input", type=str, help="Input .npz file produced by nbody.py --save")
    parser.add_argument("--primary-index", type=int, default=0, help="Central/reference body index")
    parser.add_argument("--secondary-index", type=int, default=1, help="Orbiting body index")
    parser.add_argument("--compare", type=str, default=None, help="Optional baseline .npz file to subtract")
    return parser.parse_args(argv)


def summarize(path: str | Path, *, primary_index: int, secondary_index: int) -> tuple[float, int]:
    results = load_results(path)
    peri_times, peri_positions, peri_radii = periapsis_samples(
        results["times"],
        results["traj"],
        primary_index=primary_index,
        secondary_index=secondary_index,
    )
    rate = estimate_precession_rate(
        results["times"],
        results["traj"],
        primary_index=primary_index,
        secondary_index=secondary_index,
    )
    print(path)
    print(f"  periapsis_count: {peri_times.size}")
    print(f"  first_periapsis_time: {peri_times[0]:.8f}")
    print(f"  last_periapsis_time: {peri_times[-1]:.8f}")
    print(f"  min_radius: {peri_radii.min():.12g}")
    print(f"  max_radius: {peri_radii.max():.12g}")
    print(f"  absolute_rate_rad_per_year: {rate:.12e}")
    print(f"  absolute_rate_arcsec_per_century: {arcsec_per_century_from_rad_per_year(rate):.12f}")
    print("  note: absolute rate includes baseline numerical drift; compare against a Newtonian run to isolate the PN signal")
    angle0 = np.arctan2(peri_positions[0, 1], peri_positions[0, 0])
    angle1 = np.arctan2(peri_positions[-1, 1], peri_positions[-1, 0])
    print(f"  first_last_raw_angle_rad: {angle0:.12e} -> {angle1:.12e}")
    return rate, peri_times.size


def main(argv=None):
    args = parse_args(argv)
    rate, count = summarize(
        args.input,
        primary_index=args.primary_index,
        secondary_index=args.secondary_index,
    )
    if args.compare:
        baseline_rate, baseline_count = summarize(
            args.compare,
            primary_index=args.primary_index,
            secondary_index=args.secondary_index,
        )
        diff = rate - baseline_rate
        print("comparison")
        print(f"  matched_inputs: {Path(args.input).name} vs {Path(args.compare).name}")
        print(f"  periapsis_counts: {count} vs {baseline_count}")
        print(f"  baseline_subtracted_rate_rad_per_year: {diff:.12e}")
        print(f"  baseline_subtracted_rate_arcsec_per_century: {arcsec_per_century_from_rad_per_year(diff):.12f}")
        print("  note: this baseline-subtracted rate is the physically meaningful PN estimate")


if __name__ == "__main__":
    main()
