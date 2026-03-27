from __future__ import annotations

import argparse
from pathlib import Path

from analysis import load_results, summarize_system_validation
from validation_profiles import SOLAR_SYSTEM_VALIDATION_PROFILES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Report orbital-element drift and precession validation for a saved Solar System Newtonian vs 1PN pair."
    )
    parser.add_argument("reference", type=str, help="Reference .npz file, typically Newtonian")
    parser.add_argument("candidate", type=str, help="Candidate .npz file, typically central-body 1PN")
    parser.add_argument("--primary-index", type=int, default=0, help="Primary body index, usually the Sun")
    parser.add_argument(
        "--profile",
        choices=sorted(SOLAR_SYSTEM_VALIDATION_PROFILES),
        default="inner-planets",
        help="Named validation profile used for run-planning context in the header",
    )
    return parser.parse_args(argv)


def _format_optional(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def _format_float(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.6e}"


def main(argv=None):
    args = parse_args(argv)
    profile = SOLAR_SYSTEM_VALIDATION_PROFILES[args.profile]
    reference_results = load_results(args.reference)
    candidate_results = load_results(args.candidate)
    summaries = summarize_system_validation(
        reference_results,
        candidate_results,
        primary_index=args.primary_index,
    )

    print(f"reference: {Path(args.reference).name}")
    print(f"candidate: {Path(args.candidate).name}")
    print(f"profile: {profile.name} | years={profile.years:g} | dt={profile.dt:g} | save_every={profile.save_every}")
    print(f"profile_note: {profile.note}")
    print("note: baseline-subtracted precession = candidate absolute rate minus reference absolute rate")
    print(
        "body"
        " | tolerance_status"
        " | precession_status"
        " | baseline_subtracted_arcsec/century"
        " | literature_arcsec/century"
        " | tolerance_arcsec/century"
        " | delta_a_mean"
        " | delta_e_mean"
    )
    for secondary_index in sorted(summaries):
        summary = summaries[secondary_index]
        print(
            f"{summary['body_name']}"
            f" | {summary['tolerance_status']}"
            f" | {summary.get('status', 'n/a')}"
            f" | {_format_optional(summary.get('baseline_subtracted_rate_arcsec_per_century'), '.6f')}"
            f" | {_format_optional(summary.get('literature_arcsec_per_century'), '.6f')}"
            f" | {_format_optional(summary.get('tolerance_arcsec_per_century'), '.6f')}"
            f" | {_format_float(summary['delta_semi_major_axis_mean'])}"
            f" | {_format_float(summary['delta_eccentricity_mean'])}"
        )


if __name__ == "__main__":
    main()
