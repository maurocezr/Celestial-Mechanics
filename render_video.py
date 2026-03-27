from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


Array = np.ndarray


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render an animation from a saved N-body simulation .npz file."
    )
    parser.add_argument("input", type=str, help="Input .npz file produced by nbody.py --save")
    parser.add_argument("output", type=str, help="Output animation file (.mp4 or .gif)")
    parser.add_argument("--plot-mode", choices=["2d", "3d"], default="2d", help="Animation mode")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--stride", type=int, default=1, help="Use every k-th saved frame")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    parser.add_argument("--tail", type=int, default=0, help="Show the last N frames as a trailing path (0 = full path)")
    parser.add_argument("--marker-size", type=float, default=5.0, help="Marker size for body positions")
    parser.add_argument("--line-width", type=float, default=1.2, help="Line width for trajectories")
    parser.add_argument("--title", type=str, default=None, help="Optional title override")
    return parser.parse_args(argv)


def _load_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        return value.item()
    return value


def load_results(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"times", "traj", "masses"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Input file is missing required arrays: {', '.join(missing)}")

        times = np.asarray(data["times"], dtype=np.float64)
        traj = np.asarray(data["traj"], dtype=np.float64)
        masses = np.asarray(data["masses"], dtype=np.float64)
        body_names = data["body_names"].astype("U64") if "body_names" in data else None

        return {
            "times": times,
            "traj": traj,
            "masses": masses,
            "body_names": body_names,
            "preset": _load_scalar(data, "preset", "unknown"),
            "integrator": _load_scalar(data, "integrator", "unknown"),
            "dt": float(_load_scalar(data, "dt", 0.0)),
            "note": _load_scalar(data, "note", ""),
        }


def _set_equal_3d_axes(ax, points: Array) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-12))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _compute_xy_limits(points: Array) -> tuple[tuple[float, float], tuple[float, float]]:
    mins = points[:, :2].min(axis=0)
    maxs = points[:, :2].max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    pad = 0.08 * span
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def _resolve_writer(output_path: Path):
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        if animation.writers.is_available("pillow"):
            return animation.PillowWriter
        raise RuntimeError("GIF output requires the Pillow writer, which is not available.")
    if suffix == ".mp4":
        if animation.writers.is_available("ffmpeg"):
            return animation.FFMpegWriter
        raise RuntimeError("MP4 output requires ffmpeg, which is not available on this system.")
    raise ValueError(f"Unsupported output format {suffix!r}; use .mp4 or .gif")


def _estimate_frame_bytes(fig: plt.Figure, dpi: int, num_frames: int) -> int:
    width_in, height_in = fig.get_size_inches()
    width_px = max(1, int(round(width_in * dpi)))
    height_px = max(1, int(round(height_in * dpi)))
    return width_px * height_px * 4 * num_frames


def render_video(results: dict, output_path: Path, *, plot_mode: str, fps: int, dpi: int, tail: int,
                 marker_size: float, line_width: float, title: str | None) -> None:
    traj = results["traj"]
    times = results["times"]
    masses = results["masses"]
    body_names = results["body_names"]
    num_frames, num_bodies, _ = traj.shape

    fig = plt.figure(figsize=(8, 8))
    if plot_mode == "3d":
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        ax = fig.add_subplot(1, 1, 1)

    points_all = traj.reshape(-1, 3)
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_bodies, 1)))
    lines = []
    markers = []

    for index in range(num_bodies):
        label = str(body_names[index]) if body_names is not None else f"m={masses[index]:.3g}"
        color = colors[index % len(colors)]
        if plot_mode == "3d":
            line, = ax.plot([], [], [], lw=line_width, color=color, label=label)
            marker, = ax.plot([], [], [], "o", ms=marker_size, color=color)
        else:
            line, = ax.plot([], [], lw=line_width, color=color, label=label)
            marker, = ax.plot([], [], "o", ms=marker_size, color=color)
        lines.append(line)
        markers.append(marker)

    if plot_mode == "3d":
        _set_equal_3d_axes(ax, points_all)
        ax.set_zlabel("z")
    else:
        xlim, ylim = _compute_xy_limits(points_all)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if num_bodies <= 12:
        ax.legend(loc="upper right")

    display_title = title or f"{results['preset']} | {results['integrator']} | dt={results['dt']}"
    fig.suptitle(display_title)

    def update(frame_idx: int):
        current = traj[frame_idx]
        start = 0 if tail <= 0 else max(0, frame_idx - tail + 1)
        segment = traj[start:frame_idx + 1]

        for body_idx in range(num_bodies):
            if plot_mode == "3d":
                lines[body_idx].set_data(segment[:, body_idx, 0], segment[:, body_idx, 1])
                lines[body_idx].set_3d_properties(segment[:, body_idx, 2])
                markers[body_idx].set_data([current[body_idx, 0]], [current[body_idx, 1]])
                markers[body_idx].set_3d_properties([current[body_idx, 2]])
            else:
                lines[body_idx].set_data(segment[:, body_idx, 0], segment[:, body_idx, 1])
                markers[body_idx].set_data([current[body_idx, 0]], [current[body_idx, 1]])

        ax.set_title(f"t = {times[frame_idx]:.6g}")
        return lines + markers

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=False)

    writer_cls = _resolve_writer(output_path)
    writer = writer_cls(fps=fps)
    try:
        ani.save(output_path, writer=writer, dpi=dpi)
    except MemoryError as exc:
        approx_mb = _estimate_frame_bytes(fig, dpi, num_frames) / (1024 * 1024)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            raise RuntimeError(
                "GIF export ran out of memory while Pillow was assembling frames. "
                f"This render is roughly on the order of {approx_mb:.0f} MiB of raw frame data in memory. "
                "Try one or more of: lower --dpi, increase --stride, reduce the simulation save frequency, "
                "or export to .mp4 after installing ffmpeg."
            ) from exc
        raise RuntimeError(
            f"Video export ran out of memory while writing {suffix}. "
            f"This render is roughly on the order of {approx_mb:.0f} MiB of raw frame data in memory. "
            "Try lowering --dpi or increasing --stride."
        ) from exc
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if args.fps <= 0:
        raise ValueError("--fps must be a positive integer")
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer")
    if args.tail < 0:
        raise ValueError("--tail must be zero or positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    results = load_results(input_path)

    results["times"] = results["times"][::args.stride]
    results["traj"] = results["traj"][::args.stride]

    if results["traj"].shape[0] == 0:
        raise ValueError("No frames available after applying --stride")

    render_video(
        results,
        output_path,
        plot_mode=args.plot_mode,
        fps=args.fps,
        dpi=args.dpi,
        tail=args.tail,
        marker_size=args.marker_size,
        line_width=args.line_width,
        title=args.title,
    )
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
