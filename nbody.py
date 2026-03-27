# nbody.py — Simple N-body simulator with Newtonian gravity in pure NumPy
#
# Features
# - Newtonian gravity (softened) with O(N^2) all-pairs interactions
# - Two integrators: Leapfrog (symplectic) and RK4
# - Built-in initial condition presets: 2-body Sun–Earth (AU, yr, Msun),
#   3-body (figure-8, equal masses, G=1), and star–Earth–Jupiter
# - Random N-body generator for quick experiments
# - Energy/momentum diagnostics and optional plotting
#
# Usage examples:
#   python nbody.py --preset two-body --integrator leapfrog --years 2 --dt 0.001 --plot
#   python nbody.py --preset three-body-figure8 --integrator rk4 --steps 10000 --dt 0.001 --plot
#   python nbody.py --preset star-earth-jupiter --integrator leapfrog --years 20 --dt 0.001 --plot
#   python nbody.py --preset random --n 500 --integrator leapfrog --years 2 --dt 0.001 --plot
#
# Notes
# - For astronomical units, we set G = 4π^2 so that a 1 AU circular orbit around 1 Msun
#   has period 1 year and speed ~ 2π AU/yr.
# - Softening parameter eps avoids singular forces at very short distances and improves stability.
# - Leapfrog conserves energy much better over long times than RK4 for Hamiltonian systems.

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import ssl
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from urllib import parse, request

import numpy as np
from relativity import (
    canonical_2pn_not_implemented,
    combined_1pn_acceleration,
    combined_2p5pn_acceleration,
    combined_2pn_acceleration,
    eih_energy_components,
    newtonian_acceleration,
    solve_two_body_adm_2pn_momentum_from_velocity,
    three_body_adm_hamiltonian_through_2pn,
    total_energy_2pn_two_body,
    total_energy_2pn_two_body_from_momenta,
    total_energy_1pn,
    two_body_adm_2pn_rhs,
)

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


Array = np.ndarray
AccelerationFn = Callable[[Array, Array, Array, float, float], Array]
CanonicalRHSFn = Callable[[Array, Array, Array, float, float], Tuple[Array, Array]]
CanonicalVelocityFn = Callable[[Array, Array, Array], Array]
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
DAYS_PER_YEAR = 365.25

# Masses in units of solar masses.
BUILTIN_BODY_MASSES = {
    '10': 1.0,
    'sun': 1.0,
    '199': 1.6601208254589484e-7,
    'mercury': 1.6601208254589484e-7,
    '299': 2.4478382877847715e-6,
    'venus': 2.4478382877847715e-6,
    '399': 3.00348961491547e-6,
    'earth': 3.00348961491547e-6,
    '301': 3.694303349765111e-8,
    'moon': 3.694303349765111e-8,
    '499': 3.2271560829322774e-7,
    'mars': 3.2271560829322774e-7,
    '599': 9.547919384243266e-4,
    'jupiter': 9.547919384243266e-4,
    '699': 2.8588567002946365e-4,
    'saturn': 2.8588567002946365e-4,
    '799': 4.36624961322212e-5,
    'uranus': 4.36624961322212e-5,
    '899': 5.151383772628674e-5,
    'neptune': 5.151383772628674e-5,
    '999': 6.550343146845121e-9,
    'pluto': 6.550343146845121e-9,
}
SOLAR_SYSTEM_HORIZONS_TARGETS = ['10', '199', '299', '399', '499', '599', '699', '799', '899', '999']
SOLAR_SYSTEM_BODY_NAMES = np.asarray(
    ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'],
    dtype='U64',
)


def _as_float_array(values, *, name: str, shape: Tuple[Optional[int], ...]) -> Array:
    """Convert values to a float64 array and validate its rank/shape."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions; got shape {arr.shape}")
    for dim, expected in enumerate(shape):
        if expected is not None and arr.shape[dim] != expected:
            raise ValueError(f"{name} must have shape {shape}; got {arr.shape}")
    return arr


def _normalize_body_key(value: str) -> str:
    return ''.join(ch for ch in value.strip().lower() if ch.isalnum())


def _parse_mass_overrides(specs: Optional[list[str]]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not specs:
        return overrides

    for spec in specs:
        if '=' not in spec:
            raise ValueError(f"Invalid --horizons-mass value {spec!r}; expected KEY=MASS_IN_MSUN")
        key, mass_str = spec.split('=', 1)
        mass = float(mass_str)
        if mass <= 0.0:
            raise ValueError(f"Mass override must be positive for {key!r}")
        overrides[_normalize_body_key(key)] = mass
    return overrides


def _extract_horizons_state(result_text: str) -> Tuple[Array, Array, str, str]:
    start = result_text.find('$$SOE')
    end = result_text.find('$$EOE')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Horizons response did not contain a $$SOE/$$EOE data block")

    data_block = result_text[start:end]
    lines = [line.strip() for line in data_block.splitlines() if line.strip() and not line.strip().startswith('$$')]
    if len(lines) < 3:
        raise ValueError("Horizons response did not contain a complete state vector record")

    epoch_line = lines[0]
    position_line = lines[1]
    velocity_line = lines[2]

    epoch_match = re.search(r'=\s*(.+)$', epoch_line)
    if not epoch_match:
        raise ValueError("Could not parse the Horizons epoch line")
    epoch = epoch_match.group(1).strip()

    number = r'([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)'
    pos_match = re.search(rf'X\s*=\s*{number}\s+Y\s*=\s*{number}\s+Z\s*=\s*{number}', position_line)
    vel_match = re.search(rf'VX\s*=\s*{number}\s+VY\s*=\s*{number}\s+VZ\s*=\s*{number}', velocity_line)
    if not pos_match or not vel_match:
        raise ValueError("Could not parse Horizons vector components from the response")

    positions = np.asarray([float(pos_match.group(i)) for i in range(1, 4)], dtype=np.float64)
    velocities = np.asarray([float(vel_match.group(i)) for i in range(1, 4)], dtype=np.float64)

    name_match = re.search(r'Target body name:\s*(.+?)\s+\{source:', result_text)
    target_name = name_match.group(1).strip() if name_match else 'unknown'
    return positions, velocities, target_name, epoch


def _resolve_body_mass(command: str, target_name: str, overrides: dict[str, float]) -> float:
    for key in (command, target_name):
        normalized = _normalize_body_key(key)
        if normalized in overrides:
            return overrides[normalized]
        if normalized in BUILTIN_BODY_MASSES:
            return BUILTIN_BODY_MASSES[normalized]
    raise ValueError(
        f"No mass available for Horizons target {target_name!r} ({command}). "
        "Use --horizons-mass TARGET=MASS_IN_MSUN to provide it explicitly."
    )


def fetch_horizons_ephemeris(
    commands: list[str],
    epoch: str,
    center: str,
    ref_plane: str,
    mass_specs: Optional[list[str]] = None,
    ssl_insecure: bool = False,
    G_override: Optional[float] = None,
    unit_note_override: Optional[str] = None,
    remove_com: bool = True,
) -> Tuple[Array, Array, Array, float, str, Optional[Array], Optional[str]]:
    """Fetch ephemeris state vectors from the JPL Horizons API."""
    if not commands:
        raise ValueError("At least one --horizons-target is required for Horizons ephemeris")

    mass_overrides = _parse_mass_overrides(mass_specs)
    positions = []
    velocities = []
    masses = []
    names = []
    epochs = []

    for command in commands:
        params = {
            'format': 'json',
            'COMMAND': f"'{command}'",
            'OBJ_DATA': "'YES'",
            'MAKE_EPHEM': "'YES'",
            'EPHEM_TYPE': "'VECTORS'",
            'CENTER': f"'{center}'",
            'TLIST': f"'{epoch}'",
            'OUT_UNITS': "'AU-D'",
            'REF_SYSTEM': "'ICRF'",
            'REF_PLANE': f"'{ref_plane.upper()}'",
            'VEC_TABLE': "'2'",
            'VEC_CORR': "'NONE'",
            'CSV_FORMAT': "'NO'",
        }
        url = f"{HORIZONS_API_URL}?{parse.urlencode(params)}"
        ssl_context = ssl._create_unverified_context() if ssl_insecure else None

        try:
            with request.urlopen(url, context=ssl_context) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch Horizons data for target {command!r}: {exc}") from exc

        if 'error' in payload:
            raise ValueError(f"Horizons API error for target {command!r}: {payload['error']}")
        if 'result' not in payload:
            raise ValueError(f"Horizons API response for target {command!r} did not contain 'result'")

        position_au, velocity_au_per_day, target_name, target_epoch = _extract_horizons_state(payload['result'])
        positions.append(position_au)
        velocities.append(velocity_au_per_day * DAYS_PER_YEAR)
        masses.append(_resolve_body_mass(command, target_name, mass_overrides))
        names.append(target_name)
        epochs.append(target_epoch)

    epoch_value = epochs[0] if epochs else None
    if any(ep != epoch_value for ep in epochs[1:]):
        raise ValueError("Horizons returned inconsistent epochs across targets")

    x = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    names_arr = np.asarray(names, dtype='U64')

    if remove_com:
        x, v = remove_com_motion(x, v, m)

    G = float(G_override) if G_override is not None else 4.0 * math.pi * math.pi
    unit_note = (
        f"Units: AU, yr, Msun; fetched from JPL Horizons VECTORS API; "
        f"center={center}; ref_plane={ref_plane.upper()}; G={G}"
    )
    if unit_note_override is not None:
        unit_note = unit_note_override
    if epoch_value:
        unit_note = f"{unit_note}; epoch={epoch_value}"

    return x, v, m, G, unit_note, names_arr, epoch_value


# ------------------------------ Physics Core ---------------------------------

def gravitational_acceleration(
    positions: Array,
    velocities_or_masses,
    masses_or_G,
    G_or_eps,
    eps: Optional[float] = None,
) -> Array:
    """Return Newtonian acceleration via the shared force-model interface.

    Supports both:
      gravitational_acceleration(x, v, m, G, eps)
    and the legacy API:
      gravitational_acceleration(x, m, G, eps)
    """
    if eps is None:
        velocities = np.zeros_like(positions)
        masses = np.asarray(velocities_or_masses, dtype=np.float64)
        G = float(masses_or_G)
        eps = float(G_or_eps)
    else:
        velocities = np.asarray(velocities_or_masses, dtype=np.float64)
        masses = np.asarray(masses_or_G, dtype=np.float64)
        G = float(G_or_eps)
    return newtonian_acceleration(positions, velocities, masses, G, eps)


def total_energy(positions: Array, velocities: Array, masses: Array, G: float, eps: float) -> Tuple[float, float, float]:
    """Return (K, U, E) for the system with softening eps.

    Kinetic: K = 1/2 sum m_i |v_i|^2
    Potential: U = - sum_{i<j} G m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)
    """
    # Kinetic
    K = 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))

    # Potential (pairwise, i<j)
    pos = positions
    r = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt(np.sum(r * r, axis=2) + eps * eps)
    N = pos.shape[0]
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, np.inf)
    i, j = np.triu_indices(N, k=1)
    U = -np.sum(G * masses[i] * masses[j] / dist[i, j])

    return float(K), float(U), float(K + U)


def center_of_mass(positions: Array, velocities: Array, masses: Array) -> Tuple[Array, Array]:
    """Compute center-of-mass position and velocity."""
    M = np.sum(masses)
    r_com = np.sum(positions * masses[:, None], axis=0) / M
    v_com = np.sum(velocities * masses[:, None], axis=0) / M
    return r_com, v_com


def remove_com_motion(positions: Array, velocities: Array, masses: Array) -> Tuple[Array, Array]:
    """Shift to the center-of-mass frame (positions and velocities)."""
    r_com, v_com = center_of_mass(positions, velocities, masses)
    return positions - r_com[None, :], velocities - v_com[None, :]


# ------------------------------ Integrators ----------------------------------

@dataclass
class Leapfrog:
    dt: float
    G: float
    eps: float
    masses: Array
    acceleration_fn: AccelerationFn

    def step(self, x: Array, v: Array) -> Tuple[Array, Array]:
        """Kick-Drift-Kick (velocity Verlet / leapfrog)."""
        a = self.acceleration_fn(x, v, self.masses, self.G, self.eps)
        v_half = v + 0.5 * self.dt * a
        x_new = x + self.dt * v_half
        a_new = self.acceleration_fn(x_new, v_half, self.masses, self.G, self.eps)
        v_new = v_half + 0.5 * self.dt * a_new
        return x_new, v_new


@dataclass
class RK4:
    dt: float
    G: float
    eps: float
    masses: Array
    acceleration_fn: AccelerationFn

    def step(self, x: Array, v: Array) -> Tuple[Array, Array]:
        """Classic 4th-order Runge–Kutta for second-order ODEs."""
        a1 = self.acceleration_fn(x, v, self.masses, self.G, self.eps)
        k1x, k1v = v, a1

        a2 = self.acceleration_fn(
            x + 0.5 * self.dt * k1x,
            v + 0.5 * self.dt * k1v,
            self.masses,
            self.G,
            self.eps,
        )
        k2x, k2v = v + 0.5 * self.dt * k1v, a2

        a3 = self.acceleration_fn(
            x + 0.5 * self.dt * k2x,
            v + 0.5 * self.dt * k2v,
            self.masses,
            self.G,
            self.eps,
        )
        k3x, k3v = v + 0.5 * self.dt * k2v, a3

        a4 = self.acceleration_fn(
            x + self.dt * k3x,
            v + self.dt * k3v,
            self.masses,
            self.G,
            self.eps,
        )
        k4x, k4v = v + self.dt * k3v, a4

        x_new = x + (self.dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v_new = v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return x_new, v_new


@dataclass
class CanonicalRK4:
    dt: float
    G: float
    eps: float
    masses: Array
    rhs_fn: CanonicalRHSFn

    def step(self, x: Array, p: Array) -> Tuple[Array, Array]:
        k1x, k1p = self.rhs_fn(x, p, self.masses, self.G, self.eps)

        k2x, k2p = self.rhs_fn(
            x + 0.5 * self.dt * k1x,
            p + 0.5 * self.dt * k1p,
            self.masses,
            self.G,
            self.eps,
        )

        k3x, k3p = self.rhs_fn(
            x + 0.5 * self.dt * k2x,
            p + 0.5 * self.dt * k2p,
            self.masses,
            self.G,
            self.eps,
        )

        k4x, k4p = self.rhs_fn(
            x + self.dt * k3x,
            p + self.dt * k3p,
            self.masses,
            self.G,
            self.eps,
        )

        x_new = x + (self.dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        p_new = p + (self.dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        return x_new, p_new


def canonical_velocity_from_momenta(positions: Array, momenta: Array, masses: Array) -> Array:
    del positions
    return momenta / masses[:, None]


def canonical_momenta_from_velocities(velocities: Array, masses: Array) -> Array:
    return velocities * masses[:, None]


def newtonian_canonical_rhs(
    positions: Array,
    momenta: Array,
    masses: Array,
    G: float,
    eps: float,
) -> Tuple[Array, Array]:
    velocities = canonical_velocity_from_momenta(positions, momenta, masses)
    accelerations = gravitational_acceleration(positions, velocities, masses, G, eps)
    return velocities, masses[:, None] * accelerations


def run_canonical_simulation(
    positions: Array,
    momenta: Array,
    masses: Array,
    G: float,
    cfg: "SimulationConfig",
    rhs_fn: CanonicalRHSFn,
    velocity_fn: CanonicalVelocityFn = canonical_velocity_from_momenta,
    energy_fn: Optional[Callable[[Array, Array, Array, float, float], float]] = None,
    energy_kind: str = 'canonical',
    note: str = '',
    record_energy: bool = True,
) -> dict:
    N = positions.shape[0]
    stepper = CanonicalRK4(cfg.dt, G, cfg.eps, masses, rhs_fn)

    T = cfg.steps // cfg.save_every + 1
    times = np.empty(T, dtype=np.float64)
    traj = np.empty((T, N, 3), dtype=np.float64)
    vel = np.empty((T, N, 3), dtype=np.float64)

    if record_energy:
        K_hist = np.empty(T, dtype=np.float64)
        U_hist = np.empty(T, dtype=np.float64)
        E_hist = np.empty(T, dtype=np.float64)
    else:
        K_hist = U_hist = E_hist = None

    x = positions.copy()
    p = momenta.copy()

    frame = 0
    v = velocity_fn(x, p, masses)
    times[frame] = 0.0
    traj[frame] = x
    vel[frame] = v
    if record_energy:
        if energy_fn is None:
            K_hist[frame], U_hist[frame], E_hist[frame] = total_energy(x, v, masses, G, cfg.eps)
        else:
            K_hist[frame] = np.nan
            U_hist[frame] = np.nan
            E_hist[frame] = energy_fn(x, p, masses, G, cfg.c if cfg.c is not None else 0.0)

    for step in range(1, cfg.steps + 1):
        x, p = stepper.step(x, p)
        if step % cfg.save_every == 0:
            frame += 1
            v = velocity_fn(x, p, masses)
            times[frame] = step * cfg.dt
            traj[frame] = x
            vel[frame] = v
            if record_energy:
                if energy_fn is None:
                    K_hist[frame], U_hist[frame], E_hist[frame] = total_energy(x, v, masses, G, cfg.eps)
                else:
                    K_hist[frame] = np.nan
                    U_hist[frame] = np.nan
                    E_hist[frame] = energy_fn(x, p, masses, G, cfg.c if cfg.c is not None else 0.0)

    return {
        'times': times,
        'traj': traj,
        'vel': vel,
        'energy': E_hist,
        'K': K_hist,
        'U': U_hist,
        'masses': masses,
        'G': G,
        'eps': cfg.eps,
        'dt': cfg.dt,
        'preset': cfg.preset,
        'integrator': cfg.integrator,
        'note': note,
        'body_names': cfg.body_names,
        'gravity_model': cfg.gravity_model,
        'c': cfg.c,
        'pn_scope': cfg.pn_scope,
        'pn_primary_index': cfg.pn_primary_index,
        'energy_kind': energy_kind,
    }


def numerical_hamiltonian_rhs(
    positions: Array,
    momenta: Array,
    masses: Array,
    G: float,
    eps: float,
    hamiltonian_fn: Callable[[Array, Array, Array, float], float],
) -> Tuple[Array, Array]:
    del eps
    x = np.asarray(positions, dtype=np.float64)
    p = np.asarray(momenta, dtype=np.float64)
    d_h_dp = np.empty_like(p)
    d_h_dx = np.empty_like(x)

    for a in range(x.shape[0]):
        for i in range(3):
            step_p = 1e-8 * max(1.0, abs(p[a, i]))
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[a, i] += step_p
            p_minus[a, i] -= step_p
            d_h_dp[a, i] = (hamiltonian_fn(x, p_plus, masses, G) - hamiltonian_fn(x, p_minus, masses, G)) / (2.0 * step_p)

            step_x = 1e-8 * max(1.0, abs(x[a, i]))
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[a, i] += step_x
            x_minus[a, i] -= step_x
            d_h_dx[a, i] = (hamiltonian_fn(x_plus, p, masses, G) - hamiltonian_fn(x_minus, p, masses, G)) / (2.0 * step_x)

    return d_h_dp, -d_h_dx


def run_canonical_2pn_simulation(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    cfg: "SimulationConfig",
    record_energy: bool = True,
) -> dict:
    """Canonical 2PN evolution path.

    Exact ADM two-body 2PN is implemented analytically. Milestone 1 of the
    many-body path implements the explicit three-body ADM 2PN Hamiltonian
    through numerical Hamilton equations. Broader many-body support remains
    closed until the corresponding Hamiltonian terms are added.
    """
    if cfg.c is None:
        raise ValueError("--c is required when --gravity-model 2pn is used")

    if cfg.pn_scope == 'two-body':
        if positions.shape[0] != 2:
            raise ValueError("The current canonical 2PN two-body implementation supports exactly two bodies")

        x_com, v_com = center_of_mass(positions, velocities, masses)
        positions_com, velocities_com = remove_com_motion(positions, velocities, masses)
        relative_position = positions_com[0] - positions_com[1]
        total_mass = float(np.sum(masses))
        relative_velocity = velocities_com[0] - velocities_com[1]
        relative_momentum = solve_two_body_adm_2pn_momentum_from_velocity(
            relative_position,
            relative_velocity,
            masses,
            G,
            cfg.c,
        )

        reduced_positions = np.asarray([
            x_com + (masses[1] / total_mass) * relative_position,
            x_com - (masses[0] / total_mass) * relative_position,
        ], dtype=np.float64)
        reduced_momenta = np.asarray([
            relative_momentum,
            -relative_momentum,
        ], dtype=np.float64)

        def rhs_fn(x: Array, p: Array, local_masses: Array, local_G: float, eps: float) -> Tuple[Array, Array]:
            del eps
            rel_x = x[0] - x[1]
            rel_p = p[0]
            d_rel_x, d_rel_p = two_body_adm_2pn_rhs(rel_x, rel_p, local_masses, local_G, cfg.c)
            return (
                np.asarray([
                    (local_masses[1] / np.sum(local_masses)) * d_rel_x,
                    -(local_masses[0] / np.sum(local_masses)) * d_rel_x,
                ], dtype=np.float64),
                np.asarray([d_rel_p, -d_rel_p], dtype=np.float64),
            )

        def velocity_fn(current_positions: Array, momentum: Array, local_masses: Array) -> Array:
            rel_velocity, _ = two_body_adm_2pn_rhs(
                current_positions[0] - current_positions[1],
                momentum[0],
                local_masses,
                G,
                cfg.c,
            )
            com_velocity = v_com
            total = np.sum(local_masses)
            return np.asarray([
                com_velocity + (local_masses[1] / total) * rel_velocity,
                com_velocity - (local_masses[0] / total) * rel_velocity,
            ], dtype=np.float64)

        return run_canonical_simulation(
            reduced_positions,
            reduced_momenta,
            masses,
            G,
            cfg,
            rhs_fn=rhs_fn,
            velocity_fn=velocity_fn,
            energy_fn=lambda x, p, m, local_G, local_c: total_energy_2pn_two_body_from_momenta(
                x,
                p,
                m,
                local_G,
                local_c,
            ),
            energy_kind='2pn-two-body',
            note='Canonical ADM two-body 2PN backend',
            record_energy=record_energy,
        )

    if cfg.pn_scope == 'eih' and positions.shape[0] == 3:
        x_com, v_com = center_of_mass(positions, velocities, masses)
        positions_com, velocities_com = remove_com_motion(positions, velocities, masses)
        momenta_com = canonical_momenta_from_velocities(velocities_com, masses)

        shifted_positions = positions_com + x_com[None, :]
        shifted_momenta = momenta_com + masses[:, None] * v_com[None, :]

        def hamiltonian_fn(x: Array, p: Array, local_masses: Array, local_G: float) -> float:
            return three_body_adm_hamiltonian_through_2pn(x, p, local_masses, local_G, cfg.c)

        def rhs_fn(x: Array, p: Array, local_masses: Array, local_G: float, eps: float) -> Tuple[Array, Array]:
            return numerical_hamiltonian_rhs(x, p, local_masses, local_G, eps, hamiltonian_fn)

        return run_canonical_simulation(
            shifted_positions,
            shifted_momenta,
            masses,
            G,
            cfg,
            rhs_fn=rhs_fn,
            velocity_fn=canonical_velocity_from_momenta,
            energy_fn=lambda x, p, m, local_G, local_c: three_body_adm_hamiltonian_through_2pn(
                x,
                p,
                m,
                local_G,
                local_c,
            ),
            energy_kind='2pn-adm-3body',
            note='Canonical ADM three-body 2PN backend (numerical Hamilton equations)',
            record_energy=record_energy,
        )

    del positions, velocities, masses, G, record_energy
    raise canonical_2pn_not_implemented(cfg.pn_scope)  # pragma: no cover


# ------------------------------ Presets --------------------------------------

def two_body_sun_earth() -> Tuple[Array, Array, Array, float]:
    """Sun–Earth in AU, yr, Msun units.

    Returns (positions, velocities, masses, G)
    """
    # Units: AU, yr, Msun
    G = 4.0 * math.pi * math.pi  # AU^3 / (Msun * yr^2)
    m_sun = 1.0
    m_earth = 3.00348961491547e-6  # Msun

    positions = np.array([
        [0.0, 0.0, 0.0],  # Sun at origin initially
        [1.0, 0.0, 0.0],  # Earth at 1 AU on x-axis
    ], dtype=np.float64)

    # Circular velocity for Earth around Sun: v = sqrt(G M / r) = 2π AU/yr
    v_earth = np.array([0.0, 2.0 * math.pi, 0.0])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        v_earth,
    ], dtype=np.float64)

    masses = np.array([m_sun, m_earth], dtype=np.float64)

    # Shift to center-of-mass frame (so Sun gets a small reflex motion)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def three_body_figure8() -> Tuple[Array, Array, Array, float]:
    """Three-body figure-8 choreography with equal masses and G=1.

    Initial conditions from Moore (1993)/Chenciner & Montgomery (2000).
    """
    G = 1.0
    m = 1.0
    positions = np.array([
        [-0.97000436,  0.24308753, 0.0],
        [ 0.97000436, -0.24308753, 0.0],
        [ 0.0      ,  0.0       , 0.0],
    ], dtype=np.float64)

    velocities = np.array([
        [ 0.4662036850,  0.4323657300, 0.0],
        [ 0.4662036850,  0.4323657300, 0.0],
        [-0.9324073700, -0.8647314600, 0.0],
    ], dtype=np.float64)

    masses = np.array([m, m, m], dtype=np.float64)

    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def star_earth_jupiter() -> Tuple[Array, Array, Array, float]:
    """Simple 3-body with a Sun-like star, Earth, and Jupiter in AU, yr, Msun.

    Planet initial velocities are set for circular orbits around the star,
    then the system is shifted to the barycentric frame.
    """
    G = 4.0 * math.pi * math.pi
    m_star = 1.0
    m_earth = 3.00348961491547e-6
    m_jup = 9.547919384243266e-4

    r_earth = 1.0
    r_jup = 5.2044

    def circular_velocity(M, r):
        return math.sqrt(G * M / r)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [r_earth, 0.0, 0.0],
        [r_jup, 0.0, 0.0],
    ], dtype=np.float64)

    v_e = circular_velocity(m_star, r_earth)
    v_j = circular_velocity(m_star, r_jup)

    velocities = np.array([
        [0.0, 0.0, 0.0],             # star
        [0.0, v_e, 0.0],             # Earth
        [0.0, v_j, 0.0],             # Jupiter
    ], dtype=np.float64)

    masses = np.array([m_star, m_earth, m_jup], dtype=np.float64)

    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def mercury_relativistic_demo() -> Tuple[Array, Array, Array, float]:
    """Sun-Mercury system in AU, yr, Msun units for 1PN precession experiments."""
    G = 4.0 * math.pi * math.pi
    m_sun = 1.0
    m_mercury = BUILTIN_BODY_MASSES['mercury']
    semi_major_axis = 0.387098
    eccentricity = 0.205630
    perihelion = semi_major_axis * (1.0 - eccentricity)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [perihelion, 0.0, 0.0],
    ], dtype=np.float64)

    total_mass = m_sun + m_mercury
    v_perihelion = math.sqrt(G * total_mass * (1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity)))
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, v_perihelion, 0.0],
    ], dtype=np.float64)

    masses = np.array([m_sun, m_mercury], dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def binary_pulsar_toy() -> Tuple[Array, Array, Array, float]:
    """Compact equal-mass binary in AU, yr, Msun units for stronger 1PN effects."""
    G = 4.0 * math.pi * math.pi
    m1 = 1.4
    m2 = 1.4
    semi_major_axis = 0.02
    eccentricity = 0.15
    periapsis = semi_major_axis * (1.0 - eccentricity)

    total_mass = m1 + m2
    relative_speed_periapsis = math.sqrt(
        G * total_mass * (1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity))
    )

    positions = np.array([
        [-periapsis * (m2 / total_mass), 0.0, 0.0],
        [periapsis * (m1 / total_mass), 0.0, 0.0],
    ], dtype=np.float64)
    velocities = np.array([
        [0.0, -relative_speed_periapsis * (m2 / total_mass), 0.0],
        [0.0, relative_speed_periapsis * (m1 / total_mass), 0.0],
    ], dtype=np.float64)

    masses = np.array([m1, m2], dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def psr_b1913_16() -> Tuple[Array, Array, Array, float]:
    """PSR B1913+16 in AU, yr, Msun units from published timing parameters."""
    G = 4.0 * math.pi * math.pi
    m1 = 1.438
    m2 = 1.390
    eccentricity = 0.6171338
    orbital_period_days = 0.322997462727
    orbital_period_years = orbital_period_days / DAYS_PER_YEAR
    total_mass = m1 + m2
    semi_major_axis = (total_mass * orbital_period_years * orbital_period_years) ** (1.0 / 3.0)
    periapsis = semi_major_axis * (1.0 - eccentricity)
    relative_speed_periapsis = math.sqrt(
        G * total_mass * (1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity))
    )

    positions = np.array([
        [-periapsis * (m2 / total_mass), 0.0, 0.0],
        [periapsis * (m1 / total_mass), 0.0, 0.0],
    ], dtype=np.float64)
    velocities = np.array([
        [0.0, -relative_speed_periapsis * (m2 / total_mass), 0.0],
        [0.0, relative_speed_periapsis * (m1 / total_mass), 0.0],
    ], dtype=np.float64)

    masses = np.array([m1, m2], dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def psr_b1534_12() -> Tuple[Array, Array, Array, float]:
    """PSR B1534+12 in AU, yr, Msun units from published timing parameters."""
    G = 4.0 * math.pi * math.pi
    m1 = 1.3332
    m2 = 1.3452
    eccentricity = 0.2736775
    orbital_period_days = 0.420737299122
    orbital_period_years = orbital_period_days / DAYS_PER_YEAR
    total_mass = m1 + m2
    semi_major_axis = (total_mass * orbital_period_years * orbital_period_years) ** (1.0 / 3.0)
    periapsis = semi_major_axis * (1.0 - eccentricity)
    relative_speed_periapsis = math.sqrt(
        G * total_mass * (1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity))
    )

    positions = np.array([
        [-periapsis * (m2 / total_mass), 0.0, 0.0],
        [periapsis * (m1 / total_mass), 0.0, 0.0],
    ], dtype=np.float64)
    velocities = np.array([
        [0.0, -relative_speed_periapsis * (m2 / total_mass), 0.0],
        [0.0, relative_speed_periapsis * (m1 / total_mass), 0.0],
    ], dtype=np.float64)

    masses = np.array([m1, m2], dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def psr_j0737_3039ab() -> Tuple[Array, Array, Array, float]:
    """PSR J0737-3039A/B in AU, yr, Msun units from published timing parameters."""
    G = 4.0 * math.pi * math.pi
    m1 = 1.338185
    m2 = 1.248868
    eccentricity = 0.087777023
    orbital_period_days = 0.1022515592973
    orbital_period_years = orbital_period_days / DAYS_PER_YEAR
    total_mass = m1 + m2
    semi_major_axis = (total_mass * orbital_period_years * orbital_period_years) ** (1.0 / 3.0)
    periapsis = semi_major_axis * (1.0 - eccentricity)
    relative_speed_periapsis = math.sqrt(
        G * total_mass * (1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity))
    )

    positions = np.array([
        [-periapsis * (m2 / total_mass), 0.0, 0.0],
        [periapsis * (m1 / total_mass), 0.0, 0.0],
    ], dtype=np.float64)
    velocities = np.array([
        [0.0, -relative_speed_periapsis * (m2 / total_mass), 0.0],
        [0.0, relative_speed_periapsis * (m1 / total_mass), 0.0],
    ], dtype=np.float64)

    masses = np.array([m1, m2], dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def psr_j1757_1854() -> Tuple[Array, Array, Array, float]:
    """PSR J1757-1854 in AU, yr, Msun units from published timing parameters."""
    G = 4.0 * math.pi * math.pi
    m1 = 1.3412
    m2 = 1.3917
    eccentricity = 0.6058174
    orbital_period_days = 0.183537831626
    orbital_period_years = orbital_period_days / DAYS_PER_YEAR
    total_mass = m1 + m2
    semi_major_axis = (total_mass * orbital_period_years * orbital_period_years) ** (1.0 / 3.0)
    periapsis = semi_major_axis * (1.0 - eccentricity)
    relative_speed_periapsis = math.sqrt(
        G * total_mass * (1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity))
    )

    positions = np.array([
        [-periapsis * (m2 / total_mass), 0.0, 0.0],
        [periapsis * (m1 / total_mass), 0.0, 0.0],
    ], dtype=np.float64)
    velocities = np.array([
        [0.0, -relative_speed_periapsis * (m2 / total_mass), 0.0],
        [0.0, relative_speed_periapsis * (m1 / total_mass), 0.0],
    ], dtype=np.float64)

    masses = np.array([m1, m2], dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def inner_solar_system_toy() -> Tuple[Array, Array, Array, float]:
    """Sun plus Mercury, Venus, and Earth for central-body 1PN experiments."""
    G = 4.0 * math.pi * math.pi
    bodies = [
        ("Sun", 1.0, 0.0, 0.0),
        ("Mercury", BUILTIN_BODY_MASSES["mercury"], 0.387098, 0.0),
        ("Venus", BUILTIN_BODY_MASSES["venus"], 0.723332, 2.1),
        ("Earth", BUILTIN_BODY_MASSES["earth"], 1.0, -1.2),
    ]

    positions = []
    velocities = []
    masses = []
    central_mass = bodies[0][1]
    for _, mass, radius, angle in bodies:
        if radius == 0.0:
            positions.append([0.0, 0.0, 0.0])
            velocities.append([0.0, 0.0, 0.0])
        else:
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            speed = math.sqrt(G * central_mass / radius)
            vx = -speed * math.sin(angle)
            vy = speed * math.cos(angle)
            positions.append([x, y, 0.0])
            velocities.append([vx, vy, 0.0])
        masses.append(mass)

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions, velocities, masses, G


def random_nbody(n: int, seed: int = 42, mass_spread: float = 0.0) -> Tuple[Array, Array, Array, float]:
    """Generate a random N-body system in G=1 units.

    - Positions: uniform in a unit sphere
    - Velocities: small random values ("cold" start)
    - Masses: 1 +/- mass_spread * uniform(-0.5, 0.5), clipped positive
    """
    rng = np.random.default_rng(seed)
    G = 1.0

    # Sample positions uniformly within a unit sphere
    def sample_in_sphere(N):
        vec = rng.normal(size=(N, 3))
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        radii = rng.random(N) ** (1.0 / 3.0)  # radius distribution for uniform volume
        return vec * radii[:, None]

    positions = sample_in_sphere(n)
    velocities = 0.05 * rng.normal(size=(n, 3))

    masses = np.ones(n)
    if mass_spread > 0.0:
        masses *= 1.0 + mass_spread * (rng.random(n) - 0.5)
        masses = np.clip(masses, 0.1, None)

    positions, velocities = remove_com_motion(positions, velocities, masses)
    return positions.astype(np.float64), velocities.astype(np.float64), masses.astype(np.float64), G


def load_ephemeris_json(path: Path) -> Tuple[Array, Array, Array, Optional[float], str, Optional[Array], Optional[str]]:
    """Load ephemeris ICs from a JSON file."""
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    bodies = data.get('bodies')
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("JSON ephemeris must contain a non-empty 'bodies' list")

    positions = []
    velocities = []
    masses = []
    names = []
    for idx, body in enumerate(bodies):
        if not isinstance(body, dict):
            raise ValueError(f"Body entry {idx} must be an object")
        try:
            mass = float(body['mass'])
            position = _as_float_array(body['position'], name=f"bodies[{idx}].position", shape=(3,))
            velocity = _as_float_array(body['velocity'], name=f"bodies[{idx}].velocity", shape=(3,))
        except KeyError as exc:
            raise ValueError(f"Body entry {idx} is missing required field {exc!s}") from exc

        positions.append(position)
        velocities.append(velocity)
        masses.append(mass)
        names.append(str(body.get('name', f'body-{idx}')))

    G = data.get('G')
    if G is not None:
        G = float(G)
    unit_note = str(data.get('note', 'Units: supplied by ephemeris JSON'))
    epoch = data.get('epoch')
    if epoch is not None:
        epoch = str(epoch)

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
        np.asarray(masses, dtype=np.float64),
        G,
        unit_note,
        np.asarray(names, dtype='U64'),
        epoch,
    )


def load_ephemeris_csv(path: Path) -> Tuple[Array, Array, Array, Optional[float], str, Optional[Array], Optional[str]]:
    """Load ephemeris ICs from a CSV file with one body per row."""
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"CSV ephemeris is missing required columns: {', '.join(missing)}")

        positions = []
        velocities = []
        masses = []
        names = []
        for idx, row in enumerate(reader):
            positions.append([float(row['x']), float(row['y']), float(row['z'])])
            velocities.append([float(row['vx']), float(row['vy']), float(row['vz'])])
            masses.append(float(row['mass']))
            names.append(row.get('name') or f'body-{idx}')

    if not positions:
        raise ValueError("CSV ephemeris must contain at least one body row")

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
        np.asarray(masses, dtype=np.float64),
        None,
        'Units: supplied by ephemeris CSV',
        np.asarray(names, dtype='U64'),
        None,
    )


def load_ephemeris_npz(path: Path) -> Tuple[Array, Array, Array, Optional[float], str, Optional[Array], Optional[str]]:
    """Load ephemeris ICs from an NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        if 'positions' not in data or 'velocities' not in data or 'masses' not in data:
            raise ValueError("NPZ ephemeris must contain 'positions', 'velocities', and 'masses'")

        positions = _as_float_array(data['positions'], name='positions', shape=(None, 3))
        velocities = _as_float_array(data['velocities'], name='velocities', shape=(None, 3))
        masses = _as_float_array(data['masses'], name='masses', shape=(positions.shape[0],))

        G = float(data['G']) if 'G' in data else None
        unit_note = str(data['note'].item()) if 'note' in data else 'Units: supplied by ephemeris NPZ'
        names = data['names'].astype('U64') if 'names' in data else None
        epoch = str(data['epoch'].item()) if 'epoch' in data else None

    return positions, velocities, masses, G, unit_note, names, epoch


def load_ephemeris(
    path_str: str,
    fmt: str = 'auto',
    G_override: Optional[float] = None,
    unit_note_override: Optional[str] = None,
    remove_com: bool = True,
) -> Tuple[Array, Array, Array, float, str, Optional[Array], Optional[str]]:
    """Load external ephemeris-based initial conditions."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Ephemeris file not found: {path}")

    if fmt == 'auto':
        suffix = path.suffix.lower()
        if suffix == '.json':
            fmt = 'json'
        elif suffix == '.csv':
            fmt = 'csv'
        elif suffix == '.npz':
            fmt = 'npz'
        else:
            raise ValueError(f"Could not infer ephemeris format from extension: {path.suffix}")

    if fmt == 'json':
        x, v, m, G_file, unit_note, names, epoch = load_ephemeris_json(path)
    elif fmt == 'csv':
        x, v, m, G_file, unit_note, names, epoch = load_ephemeris_csv(path)
    elif fmt == 'npz':
        x, v, m, G_file, unit_note, names, epoch = load_ephemeris_npz(path)
    else:
        raise ValueError(f"Unsupported ephemeris format: {fmt}")

    if x.shape != v.shape or x.shape[1] != 3:
        raise ValueError(f"positions and velocities must both have shape (N, 3); got {x.shape} and {v.shape}")
    if m.shape != (x.shape[0],):
        raise ValueError(f"masses must have shape ({x.shape[0]},); got {m.shape}")
    if np.any(m <= 0.0):
        raise ValueError("All masses must be strictly positive")

    G = G_override if G_override is not None else G_file
    if G is None:
        G = 1.0
        if unit_note_override is None and G_file is None:
            unit_note = f"{unit_note}; defaulted to G=1"
    else:
        G = float(G)

    if remove_com:
        x, v = remove_com_motion(x, v, m)

    if unit_note_override is not None:
        unit_note = unit_note_override

    if epoch:
        unit_note = f"{unit_note}; epoch={epoch}"

    return x, v, m, G, unit_note, names, epoch


# ------------------------------ Simulation Loop ------------------------------

@dataclass
class SimulationConfig:
    dt: float
    steps: int
    save_every: int
    eps: float
    integrator: str
    preset: str
    body_names: Optional[Array] = None
    gravity_model: str = 'newtonian'
    c: Optional[float] = None
    pn_scope: str = 'none'
    pn_primary_index: int = 0


def select_acceleration_model(cfg: SimulationConfig) -> AccelerationFn:
    if cfg.gravity_model == 'newtonian':
        return gravitational_acceleration
    if cfg.gravity_model == '1pn':
        if cfg.c is None:
            raise ValueError("--c is required when --gravity-model 1pn is used")

        def acceleration_fn(positions: Array, velocities: Array, masses: Array, G: float, eps: float) -> Array:
            return combined_1pn_acceleration(
                positions,
                velocities,
                masses,
                G,
                eps,
                cfg.c,
                scope=cfg.pn_scope,
                primary_index=cfg.pn_primary_index,
            )

        return acceleration_fn
    if cfg.gravity_model == '2.5pn':
        if cfg.c is None:
            raise ValueError("--c is required when --gravity-model 2.5pn is used")

        def acceleration_fn(positions: Array, velocities: Array, masses: Array, G: float, eps: float) -> Array:
            return combined_2p5pn_acceleration(
                positions,
                velocities,
                masses,
                G,
                eps,
                cfg.c,
                scope=cfg.pn_scope,
                primary_index=cfg.pn_primary_index,
            )

        return acceleration_fn
    if cfg.gravity_model == '2pn':
        if cfg.c is None:
            raise ValueError("--c is required when --gravity-model 2pn is used")

        def acceleration_fn(positions: Array, velocities: Array, masses: Array, G: float, eps: float) -> Array:
            return combined_2pn_acceleration(
                positions,
                velocities,
                masses,
                G,
                eps,
                cfg.c,
                scope=cfg.pn_scope,
                primary_index=cfg.pn_primary_index,
            )

        return acceleration_fn
    raise ValueError(f"Unknown gravity model: {cfg.gravity_model}")


def run_simulation(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    cfg: SimulationConfig,
    record_energy: bool = True,
) -> dict:
    """Run the N-body simulation and return a results dict with trajectories and diagnostics.

    Returns a dict with:
      - 'times': (T,) array
      - 'traj': (T, N, 3) positions
      - 'vel': (T, N, 3) velocities
      - 'energy': (T,) total energy (if record_energy)
      - 'K': (T,) kinetic energy (if record_energy)
      - 'U': (T,) potential energy (if record_energy)
    where T = number of saved frames.
    """
    if cfg.gravity_model == '2pn':
        return run_canonical_2pn_simulation(positions, velocities, masses, G, cfg, record_energy=record_energy)

    N = positions.shape[0]
    acceleration_fn = select_acceleration_model(cfg)

    # Choose integrator
    if cfg.integrator.lower() == 'leapfrog':
        stepper = Leapfrog(cfg.dt, G, cfg.eps, masses, acceleration_fn)
    elif cfg.integrator.lower() == 'rk4':
        stepper = RK4(cfg.dt, G, cfg.eps, masses, acceleration_fn)
    else:
        raise ValueError(f"Unknown integrator: {cfg.integrator}")

    T = cfg.steps // cfg.save_every + 1
    times = np.empty(T, dtype=np.float64)
    traj = np.empty((T, N, 3), dtype=np.float64)
    vel = np.empty((T, N, 3), dtype=np.float64)

    energy_kind = 'newtonian'
    if cfg.gravity_model == '1pn':
        energy_kind = '1pn'
        if cfg.pn_scope == 'eih':
            energy_kind = '1pn-eih'
    elif cfg.gravity_model == '2.5pn':
        energy_kind = '2.5pn-two-body'
    elif cfg.gravity_model == '2pn':
        energy_kind = 'unknown'

    if record_energy:
        K_hist = np.empty(T, dtype=np.float64)
        U_hist = np.empty(T, dtype=np.float64)
        E_hist = np.empty(T, dtype=np.float64)
    else:
        K_hist = U_hist = E_hist = None

    x = positions.copy()
    v = velocities.copy()

    # Initial save
    frame = 0
    times[frame] = 0.0
    traj[frame] = x
    vel[frame] = v
    if record_energy:
        K, U, E = total_energy(x, v, masses, G, cfg.eps)
        if cfg.gravity_model == '1pn' and cfg.pn_scope == 'eih':
            K, U, E = eih_energy_components(x, v, masses, G, cfg.c)
        elif cfg.gravity_model in ('1pn', '2.5pn'):
            E = total_energy_1pn(x, v, masses, G, cfg.c, scope=cfg.pn_scope, primary_index=cfg.pn_primary_index)
        K_hist[frame], U_hist[frame], E_hist[frame] = K, U, E

    # Main loop
    t = 0.0
    for n in range(1, cfg.steps + 1):
        x, v = stepper.step(x, v)
        t += cfg.dt
        if (n % cfg.save_every) == 0:
            frame += 1
            times[frame] = t
            traj[frame] = x
            vel[frame] = v
            if record_energy:
                K, U, E = total_energy(x, v, masses, G, cfg.eps)
                if cfg.gravity_model == '1pn' and cfg.pn_scope == 'eih':
                    K, U, E = eih_energy_components(x, v, masses, G, cfg.c)
                elif cfg.gravity_model in ('1pn', '2.5pn'):
                    E = total_energy_1pn(
                        x,
                        v,
                        masses,
                        G,
                        cfg.c,
                        scope=cfg.pn_scope,
                        primary_index=cfg.pn_primary_index,
                    )
                K_hist[frame], U_hist[frame], E_hist[frame] = K, U, E

    results = {
        'times': times,
        'traj': traj,
        'vel': vel,
        'K': K_hist,
        'U': U_hist,
        'energy': E_hist,
        'G': G,
        'eps': cfg.eps,
        'masses': masses,
        'integrator': cfg.integrator,
        'preset': cfg.preset,
        'dt': cfg.dt,
        'body_names': cfg.body_names,
        'gravity_model': cfg.gravity_model,
        'c': cfg.c,
        'pn_scope': cfg.pn_scope,
        'pn_primary_index': cfg.pn_primary_index,
        'energy_kind': energy_kind,
    }
    return results


# ------------------------------ Plot Helpers ---------------------------------

def _set_equal_3d_axes(ax, points: Array) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-12))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_results(res: dict, title: str = "", plot_mode: str = '2d'):
    if not _HAVE_PLT:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return

    times = res['times']
    traj = res['traj']  # (T, N, 3)
    E = res['energy']
    K = res['K']
    U = res['U']
    masses = res['masses']
    body_names = res.get('body_names')
    gravity_model = res.get('gravity_model', 'newtonian')
    energy_kind = res.get('energy_kind', 'newtonian')

    _, N, _ = traj.shape

    fig = plt.figure(figsize=(12, 5))
    if plot_mode == '3d':
        ax_orbit = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax_orbit = fig.add_subplot(1, 2, 1)
    ax_energy = fig.add_subplot(1, 2, 2)

    for j in range(N):
        label = str(body_names[j]) if body_names is not None else f"m={masses[j]:.3g}"
        if plot_mode == '3d':
            ax_orbit.plot(traj[:, j, 0], traj[:, j, 1], traj[:, j, 2], lw=1.2, label=label)
            ax_orbit.plot([traj[-1, j, 0]], [traj[-1, j, 1]], [traj[-1, j, 2]], 'o', ms=4)
        else:
            ax_orbit.plot(traj[:, j, 0], traj[:, j, 1], lw=1.2, label=label)
            ax_orbit.plot(traj[-1, j, 0], traj[-1, j, 1], 'o', ms=4)
    ax_orbit.set_xlabel('x')
    ax_orbit.set_ylabel('y')
    if plot_mode == '3d':
        ax_orbit.set_zlabel('z')
        _set_equal_3d_axes(ax_orbit, traj.reshape(-1, 3))
        ax_orbit.set_title('Trajectories (3D)')
    else:
        ax_orbit.set_aspect('equal', adjustable='box')
        ax_orbit.set_title('Trajectories (xy-plane)')
    ax_orbit.grid(True, alpha=0.3)

    if E is not None and gravity_model == 'newtonian':
        E0 = E[0]
        rel = (E - E0) / (abs(E0) + 1e-16)
        ax_energy.plot(times, rel, label='dE/E0')
        ax_energy.plot(times, K, label='K')
        ax_energy.plot(times, U, label='U')
        ax_energy.set_xlabel('time')
        ax_energy.set_ylabel('value')
        ax_energy.grid(True, alpha=0.3)
        ax_energy.legend()
        ax_energy.set_title('Energy diagnostics')
    elif E is not None and energy_kind in ('1pn', '1pn-eih', '2pn-two-body', '2.5pn-two-body'):
        E0 = E[0]
        rel = (E - E0) / (abs(E0) + 1e-16)
        if energy_kind == '2pn-two-body':
            label = 'dE/E0 (2PN total)'
        elif energy_kind == '2.5pn-two-body':
            label = 'dE/E0 (1PN bookkeeping)'
        else:
            label = 'dE/E0 (1PN total)'
        ax_energy.plot(times, rel, label=label)
        if energy_kind == '1pn-eih' and K is not None and U is not None:
            ax_energy.plot(times, K, label='K (bookkeeping)')
            ax_energy.plot(times, U, label='U (bookkeeping)')
        ax_energy.set_xlabel('time')
        ax_energy.set_ylabel('value' if energy_kind == '1pn-eih' else 'relative drift')
        ax_energy.grid(True, alpha=0.3)
        ax_energy.legend()
        if energy_kind == '1pn-eih':
            title = 'EIH 1PN Energy Diagnostic'
        elif energy_kind == '2pn-two-body':
            title = 'ADM 2PN Energy Diagnostic'
        elif energy_kind == '2.5pn-two-body':
            title = '2.5PN Inspiral Diagnostic'
        else:
            title = '1PN Energy Diagnostic'
        ax_energy.set_title(title)
        ax_energy.text(
            0.5,
            -0.18,
            (
                "EIH K/U are a diagnostic bookkeeping split; only the total 1PN energy is unambiguous."
                if energy_kind == '1pn-eih'
                else (
                    "Canonical ADM 2PN is currently implemented only for the exact two-body scope."
                    if energy_kind == '2pn-two-body'
                    else (
                        "2.5PN is dissipative: this panel shows the drift of the 1PN conservative bookkeeping energy."
                        if energy_kind == '2.5pn-two-body'
                        else "Baseline-subtracted orbital comparisons are available via analyze_precession.py and compare_orbits.py."
                    )
                )
            ),
            ha='center',
            va='top',
            transform=ax_energy.transAxes,
        )
    else:
        ax_energy.axis('off')

    suptitle = title or f"{res['preset']} • {res['integrator']} • dt={res['dt']}"
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


# ------------------------------ CLI Interface --------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="N-body simulator (Newtonian gravity) with Leapfrog and RK4 integrators.")

    presets = [
        'two-body',
        'three-body-figure8',
        'star-earth-jupiter',
        'random',
        'ephemeris',
        'solar-system-horizons',
        'mercury-relativistic',
        'binary-pulsar-toy',
        'psr-b1913+16',
        'psr-b1534+12',
        'psr-j0737-3039ab',
        'psr-j1757-1854',
        'inner-solar-system-toy',
    ]
    integrators = ['leapfrog', 'rk4']

    p.add_argument('--preset', choices=presets, default='two-body', help='Initial condition preset')
    p.add_argument('--integrator', choices=integrators, default='leapfrog', help='Time integrator')

    # Time settings: you can specify either --steps or --years; --steps has priority
    p.add_argument('--steps', type=int, default=None, help='Number of integration steps (overrides --years if provided)')
    p.add_argument('--years', type=float, default=1.0, help='Total simulated time for AU/yr presets (ignored for G=1 unless units are interpreted as arbitrary)')
    p.add_argument('--dt', type=float, default=0.001, help='Time step')
    p.add_argument('--save-every', type=int, default=10, help='Save every k steps')

    p.add_argument('--eps', type=float, default=1e-3, help='Gravitational softening length (in position units)')
    p.add_argument('--gravity-model', choices=['newtonian', '1pn', '2pn', '2.5pn'], default='newtonian', help='Acceleration model')
    p.add_argument('--c', type=float, default=None, help='Effective speed of light in simulation units for PN runs')
    p.add_argument('--pn-scope', choices=['none', 'two-body', 'central-body', 'eih'], default='none', help='Scope restriction for PN runs')
    p.add_argument('--pn-primary-index', type=int, default=0, help='Primary body index for central-body PN runs')

    # Random N-body options
    p.add_argument('--n', type=int, default=100, help='Number of bodies for --preset random')
    p.add_argument('--seed', type=int, default=42, help='Random seed for --preset random')
    p.add_argument('--mass-spread', type=float, default=0.0, help='Fractional mass spread for --preset random (0 = equal masses)')

    # Ephemeris options
    p.add_argument('--ephemeris-source', choices=['file', 'horizons'], default='file', help='Ephemeris source for --preset ephemeris')
    p.add_argument('--ephemeris-file', type=str, default=None, help='Path to ephemeris IC file (.json, .csv, or .npz) for --preset ephemeris')
    p.add_argument('--ephemeris-format', choices=['auto', 'json', 'csv', 'npz'], default='auto', help='Ephemeris IC file format')
    p.add_argument('--ephemeris-g', type=float, default=None, help='Override gravitational constant for --preset ephemeris')
    p.add_argument('--ephemeris-note', type=str, default=None, help='Override unit/metadata note for --preset ephemeris')
    p.add_argument('--horizons-target', action='append', default=None, help='Horizons target COMMAND value; repeat for multiple bodies, e.g. --horizons-target 10 --horizons-target 399')
    p.add_argument('--horizons-epoch', type=str, default=None, help='Single Horizons epoch, passed via TLIST, e.g. 2026-01-01 00:00')
    p.add_argument('--horizons-center', type=str, default='500@0', help='Horizons center code, default is Solar System barycenter')
    p.add_argument('--horizons-ref-plane', choices=['ecliptic', 'frame'], default='ecliptic', help='Horizons output reference plane')
    p.add_argument('--horizons-mass', action='append', default=None, help='Mass override in solar masses for a Horizons target, e.g. Earth=3.00348961491547e-6')
    p.add_argument('--horizons-ssl-insecure', action='store_true', help='Disable TLS certificate verification for Horizons requests (only for trusted proxy environments)')
    p.add_argument('--ephemeris-no-com', dest='ephemeris_remove_com', action='store_false', help='Keep ephemeris ICs in the supplied frame instead of shifting to the COM frame')
    p.set_defaults(ephemeris_remove_com=True)

    # Output/plot
    p.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plotting even if matplotlib is available')
    p.add_argument('--plot', dest='plot', action='store_true', help='Enable plotting (default: on if matplotlib installed)')
    p.add_argument('--plot-mode', choices=['2d', '3d'], default='2d', help='Trajectory plot mode')
    p.set_defaults(plot=True)

    p.add_argument('--save', type=str, default=None, help='Path to save results as .npz (times, traj, vel, energies, metadata)')

    args = p.parse_args(argv)
    return args


def load_preset(
    name: str,
    n: int,
    seed: int,
    mass_spread: float,
    ephemeris_source: str = 'file',
    ephemeris_file: Optional[str] = None,
    ephemeris_format: str = 'auto',
    ephemeris_g: Optional[float] = None,
    ephemeris_note: Optional[str] = None,
    horizons_targets: Optional[list[str]] = None,
    horizons_epoch: Optional[str] = None,
    horizons_center: str = '500@0',
    horizons_ref_plane: str = 'ecliptic',
    horizons_mass: Optional[list[str]] = None,
    horizons_ssl_insecure: bool = False,
    ephemeris_remove_com: bool = True,
) -> Tuple[Array, Array, Array, float, str, Optional[Array], Optional[str]]:
    body_names = None
    epoch = None
    if name == 'two-body':
        x, v, m, G = two_body_sun_earth()
        unit_note = "Units: AU, yr, Msun; G=4π²"
    elif name == 'solar-system-horizons':
        if not horizons_epoch:
            raise ValueError("--horizons-epoch is required when --preset solar-system-horizons is used")
        x, v, m, G, unit_note, _, epoch = fetch_horizons_ephemeris(
            commands=SOLAR_SYSTEM_HORIZONS_TARGETS,
            epoch=horizons_epoch,
            center=horizons_center,
            ref_plane=horizons_ref_plane,
            mass_specs=horizons_mass,
            ssl_insecure=horizons_ssl_insecure,
            G_override=ephemeris_g,
            unit_note_override=ephemeris_note,
            remove_com=ephemeris_remove_com,
        )
        body_names = SOLAR_SYSTEM_BODY_NAMES.copy()
        unit_note = f"{unit_note}; preset=solar-system-horizons"
    elif name == 'mercury-relativistic':
        x, v, m, G = mercury_relativistic_demo()
        body_names = np.asarray(['Sun', 'Mercury'], dtype='U64')
        unit_note = "Units: AU, yr, Msun; G=4*pi^2; Mercury perihelion demo"
    elif name == 'binary-pulsar-toy':
        x, v, m, G = binary_pulsar_toy()
        body_names = np.asarray(['Pulsar A', 'Pulsar B'], dtype='U64')
        unit_note = "Units: AU, yr, Msun; G=4*pi^2; compact binary toy for 1PN comparison"
    elif name == 'psr-b1913+16':
        x, v, m, G = psr_b1913_16()
        body_names = np.asarray(['PSR B1913+16 A', 'PSR B1913+16 B'], dtype='U64')
        unit_note = (
            "Units: AU, yr, Msun; G=4*pi^2; PSR B1913+16 from published masses, period, and eccentricity"
        )
    elif name == 'psr-b1534+12':
        x, v, m, G = psr_b1534_12()
        body_names = np.asarray(['PSR B1534+12 A', 'PSR B1534+12 B'], dtype='U64')
        unit_note = (
            "Units: AU, yr, Msun; G=4*pi^2; PSR B1534+12 from published masses, period, and eccentricity"
        )
    elif name == 'psr-j0737-3039ab':
        x, v, m, G = psr_j0737_3039ab()
        body_names = np.asarray(['PSR J0737-3039A', 'PSR J0737-3039B'], dtype='U64')
        unit_note = (
            "Units: AU, yr, Msun; G=4*pi^2; PSR J0737-3039A/B from published masses, period, and eccentricity"
        )
    elif name == 'psr-j1757-1854':
        x, v, m, G = psr_j1757_1854()
        body_names = np.asarray(['PSR J1757-1854', 'PSR J1757-1854 companion'], dtype='U64')
        unit_note = (
            "Units: AU, yr, Msun; G=4*pi^2; PSR J1757-1854 from published masses, period, and eccentricity"
        )
    elif name == 'inner-solar-system-toy':
        x, v, m, G = inner_solar_system_toy()
        body_names = np.asarray(['Sun', 'Mercury', 'Venus', 'Earth'], dtype='U64')
        unit_note = "Units: AU, yr, Msun; G=4*pi^2; Sun plus Mercury/Venus/Earth for central-body 1PN"
    elif name == 'three-body-figure8':
        x, v, m, G = three_body_figure8()
        unit_note = "Units: G=1 (dimensionless figure-8 ICs)"
    elif name == 'star-earth-jupiter':
        x, v, m, G = star_earth_jupiter()
        unit_note = "Units: AU, yr, Msun; G=4π²"
    elif name == 'random':
        x, v, m, G = random_nbody(n=n, seed=seed, mass_spread=mass_spread)
        unit_note = "Units: G=1 (random ICs)"
    elif name == 'ephemeris':
        if ephemeris_source == 'file':
            if not ephemeris_file:
                raise ValueError("--ephemeris-file is required when --preset ephemeris --ephemeris-source file is used")
            x, v, m, G, unit_note, body_names, epoch = load_ephemeris(
                ephemeris_file,
                fmt=ephemeris_format,
                G_override=ephemeris_g,
                unit_note_override=ephemeris_note,
                remove_com=ephemeris_remove_com,
            )
        else:
            if not horizons_epoch:
                raise ValueError("--horizons-epoch is required when --ephemeris-source horizons is used")
            x, v, m, G, unit_note, body_names, epoch = fetch_horizons_ephemeris(
                commands=horizons_targets or [],
                epoch=horizons_epoch,
                center=horizons_center,
                ref_plane=horizons_ref_plane,
                mass_specs=horizons_mass,
                ssl_insecure=horizons_ssl_insecure,
                G_override=ephemeris_g,
                unit_note_override=ephemeris_note,
                remove_com=ephemeris_remove_com,
            )
    else:
        raise ValueError(f"Unknown preset {name}")
    return x, v, m, G, unit_note, body_names, epoch


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.gravity_model != 'newtonian':
        if args.integrator != 'rk4':
            raise ValueError("Post-Newtonian runs currently require --integrator rk4")
        if args.c is None:
            raise ValueError("--c is required when --gravity-model is 1pn, 2pn, or 2.5pn")
        if args.pn_scope == 'none':
            raise ValueError("--pn-scope must be set to two-body, central-body, or eih for Post-Newtonian runs")
    else:
        if args.pn_scope != 'none':
            raise ValueError("--pn-scope is only valid with --gravity-model 1pn, 2pn, or 2.5pn")
        if args.c is not None:
            raise ValueError("--c is only valid with --gravity-model 1pn, 2pn, or 2.5pn")


def build_simulation_config(args: argparse.Namespace, body_names: Optional[Array], steps: int) -> SimulationConfig:
    return SimulationConfig(
        dt=float(args.dt),
        steps=steps,
        save_every=int(args.save_every),
        eps=float(args.eps),
        integrator=args.integrator,
        preset=args.preset,
        body_names=body_names,
        gravity_model=args.gravity_model,
        c=float(args.c) if args.c is not None else None,
        pn_scope=args.pn_scope,
        pn_primary_index=int(args.pn_primary_index),
    )


def main(argv=None):
    args = parse_args(argv)
    validate_runtime_args(args)

    # Load initial conditions
    positions, velocities, masses, G, unit_note, body_names, epoch = load_preset(
        args.preset,
        args.n,
        args.seed,
        args.mass_spread,
        ephemeris_source=args.ephemeris_source,
        ephemeris_file=args.ephemeris_file,
        ephemeris_format=args.ephemeris_format,
        ephemeris_g=args.ephemeris_g,
        ephemeris_note=args.ephemeris_note,
        horizons_targets=args.horizons_target,
        horizons_epoch=args.horizons_epoch,
        horizons_center=args.horizons_center,
        horizons_ref_plane=args.horizons_ref_plane,
        horizons_mass=args.horizons_mass,
        horizons_ssl_insecure=args.horizons_ssl_insecure,
        ephemeris_remove_com=args.ephemeris_remove_com,
    )

    # Determine number of steps
    if args.steps is not None:
        steps = args.steps
    else:
        # Interpret --years as total time; steps = years / dt
        total_time = float(args.years)
        steps = max(1, int(round(total_time / args.dt)))

    cfg = build_simulation_config(args, body_names, steps)

    # Run
    results = run_simulation(positions, velocities, masses, G, cfg, record_energy=True)

    # Save
    if args.save:
        out = {
            'times': results['times'],
            'traj': results['traj'],
            'vel': results['vel'],
            'K': results['K'],
            'U': results['U'],
            'energy': results['energy'],
            'G': np.array(results['G']),
            'eps': np.array(results['eps']),
            'masses': results['masses'],
            'integrator': np.array(results['integrator']),
            'preset': np.array(results['preset']),
            'dt': np.array(results['dt']),
            'gravity_model': np.array(results['gravity_model']),
            'pn_scope': np.array(results['pn_scope']),
            'pn_primary_index': np.array(results['pn_primary_index']),
            'energy_kind': np.array(results['energy_kind']),
            'note': np.array(unit_note),
        }
        if results['c'] is not None:
            out['c'] = np.array(results['c'])
        if results.get('body_names') is not None:
            out['body_names'] = results['body_names']
        if epoch is not None:
            out['epoch'] = np.array(epoch)
        np.savez(args.save, **out)
        print(f"Saved results to {args.save}")

    # Plot
    if args.plot:
        title = f"{args.preset} | {args.integrator} | dt={args.dt} | eps={args.eps} | {unit_note}"
        plot_results(results, title, plot_mode=args.plot_mode)


if __name__ == '__main__':
    main()
