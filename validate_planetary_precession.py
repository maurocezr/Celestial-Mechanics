from __future__ import annotations

import argparse
from pathlib import Path

from analysis import load_results, summarize_system_precession_vs_references


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare baseline-subtracted perihelion precession against bundled planetary reference values."
    )
    parser.add_argument("reference", type=str, help="Reference .npz file, typically Newtonian")
    parser.add_argument("candidate", type=str, help="Candidate .npz file, typically central-body 1PN")
    parser.add_argument("--primary-index", type=int, default=0, help="Primary body index, usually the Sun")
    return parser.parse_args(argv)


def _format_optional(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def main(argv=None):
    args = parse_args(argv)
    reference_results = load_results(args.reference)
    candidate_results = load_results(args.candidate)
    summaries = summarize_system_precession_vs_references(
        reference_results,
        candidate_results,
        primary_index=args.primary_index,
    )

    print(f"reference: {Path(args.reference).name}")
    print(f"candidate: {Path(args.candidate).name}")
    print("note: baseline-subtracted rate = candidate absolute rate minus reference absolute rate")
    print(
        "body"
        " | status"
        " | baseline_subtracted_arcsec/century"
        " | literature_arcsec/century"
        " | literature_error"
        " | literature_rel_error_%"
        " | formula_arcsec/century"
        " | formula_error"
    )

    for secondary_index in sorted(summaries):
        summary = summaries[secondary_index]
        if summary["status"] != "ok":
            print(
                f"{summary['body_name']}"
                f" | {summary['status']}"
                " | n/a | n/a | n/a | n/a | n/a | n/a"
            )
            continue
        print(
            f"{summary['body_name']}"
            f" | {summary['status']}"
            f" | {summary['baseline_subtracted_rate_arcsec_per_century']:.6f}"
            f" | {_format_optional(summary.get('literature_arcsec_per_century'), '.6f')}"
            f" | {_format_optional(summary.get('literature_error_arcsec_per_century'), '.6f')}"
            f" | {_format_optional(summary.get('literature_relative_error_percent'), '.3f')}"
            f" | {_format_optional(summary.get('formula_arcsec_per_century'), '.6f')}"
            f" | {_format_optional(summary.get('formula_error_arcsec_per_century'), '.6f')}"
        )


if __name__ == "__main__":
    main()
