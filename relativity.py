from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
PNScope = Literal["two-body", "central-body", "eih"]


@dataclass(frozen=True)
class PNConfig:
    order: int
    c: float
    scope: PNScope = "two-body"


_TWO_PN_REFERENCE_NOTE = (
    "The conservative many-body 2PN problem is naturally formulated through an "
    "ADM/Fokker Hamiltonian with canonical variables and genuine many-body terms, "
    "not as the explicit x,v -> a(x,v) force law used by the Newtonian/1PN paths."
)


def newtonian_acceleration(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
) -> Array:
    """Return Newtonian all-pairs acceleration with Plummer-like softening."""
    del velocities
    r = positions[None, :, :] - positions[:, None, :]
    dist2 = np.sum(r * r, axis=2) + eps * eps
    np.fill_diagonal(dist2, np.inf)
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    return G * np.sum(masses[None, :, None] * r * inv_dist3[:, :, None], axis=1)


def validate_pn_inputs(
    positions: Array,
    velocities: Array,
    masses: Array,
    scope: PNScope,
    primary_index: int | None = None,
) -> None:
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions and velocities must both have shape (N, 3); got {positions.shape} and {velocities.shape}"
        )
    if masses.shape != (positions.shape[0],):
        raise ValueError(f"masses must have shape ({positions.shape[0]},); got {masses.shape}")
    if np.any(masses <= 0.0):
        raise ValueError("All masses must be strictly positive for Post-Newtonian corrections")
    if scope not in ("two-body", "central-body", "eih"):
        raise ValueError(f"Unsupported PN scope: {scope}")
    if scope == "two-body" and positions.shape[0] != 2:
        raise ValueError(
            "The current 1PN implementation only supports exactly two bodies."
        )
    if scope == "central-body":
        if positions.shape[0] < 2:
            raise ValueError("The central-body 1PN approximation requires at least two bodies")
        if primary_index is None:
            raise ValueError("primary_index is required for central-body 1PN")
        validate_central_body_primary(masses, primary_index)
    if scope == "eih" and positions.shape[0] < 2:
        raise ValueError("The EIH 1PN implementation requires at least two bodies")


def validate_2pn_inputs(
    positions: Array,
    velocities: Array,
    masses: Array,
    scope: PNScope,
    primary_index: int | None = None,
) -> None:
    validate_pn_inputs(positions, velocities, masses, scope, primary_index=primary_index)


def validate_2p5pn_inputs(
    positions: Array,
    velocities: Array,
    masses: Array,
    scope: PNScope,
    primary_index: int | None = None,
) -> None:
    validate_pn_inputs(positions, velocities, masses, scope, primary_index=primary_index)
    if scope != "two-body":
        raise NotImplementedError(
            "The current 2.5PN implementation is limited to the two-body radiation-reaction problem."
        )


def canonical_2pn_backend_name(scope: PNScope) -> str:
    if scope == "two-body":
        return "adm-hamiltonian-2pn-two-body"
    if scope == "eih":
        return "adm-hamiltonian-2pn-nbody"
    raise NotImplementedError(
        "2PN is not defined for the current central-body approximation. "
        "Use a canonical ADM-style two-body or full many-body Hamiltonian backend instead."
    )


def canonical_2pn_not_implemented(scope: PNScope) -> NotImplementedError:
    backend = canonical_2pn_backend_name(scope)
    return NotImplementedError(
        f"2PN backend '{backend}' is not implemented yet. {_TWO_PN_REFERENCE_NOTE}"
    )


def validate_central_body_primary(
    masses: Array,
    primary_index: int,
) -> None:
    if not (0 <= primary_index < masses.shape[0]):
        raise ValueError(f"primary_index {primary_index} is out of bounds for {masses.shape[0]} bodies")


def relative_state_to_primary(
    positions: Array,
    velocities: Array,
    primary_index: int,
) -> tuple[Array, Array]:
    primary_position = positions[primary_index]
    primary_velocity = velocities[primary_index]
    return positions - primary_position[None, :], velocities - primary_velocity[None, :]


def _pairwise_geometry(positions: Array, eps: float) -> tuple[Array, Array, Array, Array]:
    diff = positions[None, :, :] - positions[:, None, :]
    dist2 = np.sum(diff * diff, axis=2) + eps * eps
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    inv_dist = 1.0 / dist
    inv_dist2 = inv_dist * inv_dist
    unit_ba = diff * inv_dist[:, :, None]
    return diff, dist, inv_dist, unit_ba


def post_newtonian_1pn_acceleration(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
    scope: PNScope = "two-body",
    primary_index: int = 0,
) -> Array:
    """Return the 1PN correction for the current limited two-body implementation.

    The correction is based on the standard Schwarzschild-like 1PN relative
    acceleration form. It is intended as a bridge feature for Mercury-like
    precession and compact-binary toy problems, not as a general arbitrary-N
    PN solver.
    """
    validate_pn_inputs(positions, velocities, masses, scope, primary_index=primary_index)
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 1PN corrections")

    if scope == "central-body":
        return central_body_1pn_correction(positions, velocities, masses, G, eps, c, primary_index=primary_index)
    if scope == "eih":
        return eih_1pn_correction(positions, velocities, masses, G, eps, c)

    del eps

    m1, m2 = masses
    total_mass = m1 + m2
    if total_mass <= 0.0:
        raise ValueError("Total system mass must be positive")

    rel_pos = positions[1] - positions[0]
    rel_vel = velocities[1] - velocities[0]
    radius = float(np.linalg.norm(rel_pos))
    if radius == 0.0:
        raise ValueError("1PN correction is undefined for zero separation")

    radial_velocity = float(np.dot(rel_pos, rel_vel)) / radius
    speed2 = float(np.dot(rel_vel, rel_vel))
    mu = G * total_mass
    eta = (m1 * m2) / (total_mass * total_mass)

    # Relative 1PN correction for a comparable-mass binary in harmonic coordinates.
    coeff = mu / (c * c * radius**3)
    rel_correction = coeff * (
        (
            (4.0 + 2.0 * eta) * mu / radius
            - (1.0 + 3.0 * eta) * speed2
            + 1.5 * eta * radial_velocity * radial_velocity
        )
        * rel_pos
        + (4.0 - 2.0 * eta) * radial_velocity * radius * rel_vel
    )

    # Convert the relative correction into barycentric accelerations that preserve
    # the center-of-mass relation m1*a1 + m2*a2 = 0.
    return np.asarray(
        [
            -(m2 / total_mass) * rel_correction,
            (m1 / total_mass) * rel_correction,
        ],
        dtype=np.float64,
    )


def central_body_1pn_correction(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
    primary_index: int = 0,
) -> Array:
    """Return a central-body 1PN correction for a dominant relativistic source.

    This approximation applies a Schwarzschild-like 1PN correction between the
    designated primary and each secondary body independently. Newtonian mutual
    interactions between all bodies remain in the baseline acceleration.
    """
    del eps
    validate_pn_inputs(positions, velocities, masses, "central-body", primary_index=primary_index)
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 1PN corrections")

    corrections = np.zeros_like(positions, dtype=np.float64)
    rel_positions, rel_velocities = relative_state_to_primary(positions, velocities, primary_index)
    primary_mass = float(masses[primary_index])

    for secondary_index in range(positions.shape[0]):
        if secondary_index == primary_index:
            continue

        secondary_mass = float(masses[secondary_index])
        total_mass = primary_mass + secondary_mass
        rel_pos = rel_positions[secondary_index]
        rel_vel = rel_velocities[secondary_index]
        radius = float(np.linalg.norm(rel_pos))
        if radius == 0.0:
            raise ValueError("central-body 1PN correction is undefined for zero separation")

        radial_velocity = float(np.dot(rel_pos, rel_vel)) / radius
        speed2 = float(np.dot(rel_vel, rel_vel))
        mu = G * total_mass
        coeff = mu / (c * c * radius**3)
        rel_correction = coeff * (
            (4.0 * mu / radius - speed2) * rel_pos + 4.0 * radial_velocity * radius * rel_vel
        )

        corrections[primary_index] += -(secondary_mass / total_mass) * rel_correction
        corrections[secondary_index] += (primary_mass / total_mass) * rel_correction

    return corrections


def eih_1pn_correction(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
) -> Array:
    """Return the harmonic-coordinate EIH 1PN correction for arbitrary N.

    The exact EIH equations are acceleration-dependent. At 1PN order, replacing
    the implicit accelerations on the right-hand side with Newtonian
    accelerations preserves the formal 1PN accuracy while keeping the update
    explicit for the integrator.
    """
    validate_pn_inputs(positions, velocities, masses, "eih")
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 1PN corrections")

    _, _, inv_dist, unit_ba = _pairwise_geometry(positions, eps)
    inv_dist2 = inv_dist * inv_dist
    n_ab = -unit_ba

    v2 = np.sum(velocities * velocities, axis=1)
    vv = velocities @ velocities.T
    newtonian = newtonian_acceleration(positions, velocities, masses, G, eps)

    # U_A = sum_{C != A} G m_C / r_AC
    pair_mass_over_r = G * masses[None, :] * inv_dist
    potential_sum = np.sum(pair_mass_over_r, axis=1)

    n_ab_dot_vb = np.sum(n_ab * velocities[None, :, :], axis=2)
    xb_minus_xa_dot_ab = np.sum((positions[None, :, :] - positions[:, None, :]) * newtonian[None, :, :], axis=2)
    scalar_term = (
        v2[:, None]
        + 2.0 * v2[None, :]
        - 4.0 * vv
        - 1.5 * n_ab_dot_vb * n_ab_dot_vb
        - 4.0 * potential_sum[:, None]
        - potential_sum[None, :]
        + 0.5 * xb_minus_xa_dot_ab
    )
    radial_term = np.sum(n_ab * (4.0 * velocities[:, None, :] - 3.0 * velocities[None, :, :]), axis=2)
    relative_velocity = velocities[:, None, :] - velocities[None, :, :]

    pair_prefactor = G * masses[None, :] * inv_dist2
    term1 = np.sum(pair_prefactor[:, :, None] * unit_ba * scalar_term[:, :, None], axis=1)
    term2 = np.sum(pair_prefactor[:, :, None] * radial_term[:, :, None] * relative_velocity, axis=1)
    term3 = 3.5 * np.sum((G * masses[None, :] * inv_dist)[:, :, None] * newtonian[None, :, :], axis=1)
    return (term1 + term2 + term3) / (c * c)


def combined_1pn_acceleration(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
    scope: PNScope = "two-body",
    primary_index: int = 0,
) -> Array:
    return newtonian_acceleration(positions, velocities, masses, G, eps) + post_newtonian_1pn_acceleration(
        positions, velocities, masses, G, eps, c, scope, primary_index=primary_index
    )


def post_newtonian_2p5pn_acceleration(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
    scope: PNScope = "two-body",
    primary_index: int = 0,
) -> Array:
    """Return the leading two-body 2.5PN radiation-reaction acceleration.

    This implements the standard leading-order dissipative relative correction
    in the center-of-mass frame. The current implementation is intentionally
    limited to the two-body problem and is meant to be combined with the
    Newtonian + two-body 1PN conservative terms.
    """
    del eps, primary_index
    validate_2p5pn_inputs(positions, velocities, masses, scope)
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 2.5PN corrections")

    m1, m2 = masses
    total_mass = m1 + m2
    eta = (m1 * m2) / (total_mass * total_mass)
    rel_pos = positions[1] - positions[0]
    rel_vel = velocities[1] - velocities[0]
    radius = float(np.linalg.norm(rel_pos))
    if radius == 0.0:
        raise ValueError("2.5PN correction is undefined for zero separation")

    rhat = rel_pos / radius
    radial_velocity = float(np.dot(rel_pos, rel_vel)) / radius
    speed2 = float(np.dot(rel_vel, rel_vel))
    gm = G * total_mass

    coeff = (8.0 / 5.0) * eta * gm * gm / (c**5 * radius**3)
    rel_correction = coeff * (
        radial_velocity * (3.0 * speed2 + (17.0 / 3.0) * gm / radius) * rhat
        - (speed2 + 3.0 * gm / radius) * rel_vel
    )
    return np.asarray(
        [
            -(m2 / total_mass) * rel_correction,
            (m1 / total_mass) * rel_correction,
        ],
        dtype=np.float64,
    )


def combined_2p5pn_acceleration(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
    scope: PNScope = "two-body",
    primary_index: int = 0,
) -> Array:
    validate_2p5pn_inputs(positions, velocities, masses, scope, primary_index=primary_index)
    return combined_1pn_acceleration(
        positions,
        velocities,
        masses,
        G,
        eps,
        c,
        scope=scope,
        primary_index=primary_index,
    ) + post_newtonian_2p5pn_acceleration(
        positions,
        velocities,
        masses,
        G,
        eps,
        c,
        scope=scope,
        primary_index=primary_index,
    )


def total_energy_1pn(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    c: float,
    scope: PNScope = "two-body",
    primary_index: int = 0,
) -> float:
    """Return the standard weak-field 1PN two-body energy.

    This is the conserved energy associated with the current two-body 1PN model
    in the weak-field limit. It is intended for diagnostics and comparison, not
    as a general arbitrary-N PN energy.
    """
    validate_pn_inputs(positions, velocities, masses, scope, primary_index=primary_index)
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 1PN energy")

    if scope == "eih":
        _, _, total = eih_energy_components(positions, velocities, masses, G, c)
        return total

    if scope == "central-body":
        total_energy = 0.0
        rel_positions, rel_velocities = relative_state_to_primary(positions, velocities, primary_index)
        primary_mass = float(masses[primary_index])
        for secondary_index in range(positions.shape[0]):
            if secondary_index == primary_index:
                continue
            pair_positions = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    rel_positions[secondary_index],
                ],
                dtype=np.float64,
            )
            pair_velocities = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    rel_velocities[secondary_index],
                ],
                dtype=np.float64,
            )
            pair_masses = np.asarray([primary_mass, masses[secondary_index]], dtype=np.float64)
            total_energy += total_energy_1pn(pair_positions, pair_velocities, pair_masses, G, c, scope="two-body")
        return float(total_energy)

    m1, m2 = masses
    total_mass = m1 + m2
    eta = (m1 * m2) / (total_mass * total_mass)
    mu = (m1 * m2) / total_mass
    rel_pos = positions[1] - positions[0]
    rel_vel = velocities[1] - velocities[0]
    radius = float(np.linalg.norm(rel_pos))
    if radius == 0.0:
        raise ValueError("1PN energy is undefined for zero separation")

    speed2 = float(np.dot(rel_vel, rel_vel))
    radial_velocity = float(np.dot(rel_pos, rel_vel)) / radius
    gm = G * total_mass

    newtonian = mu * (0.5 * speed2 - gm / radius)
    correction = (
        0.375 * (1.0 - 3.0 * eta) * speed2 * speed2
        + 0.5 * (gm / radius) * ((3.0 + eta) * speed2 + eta * radial_velocity * radial_velocity + gm / radius)
    )
    return float(newtonian + (mu / (c * c)) * correction)


def two_body_adm_2pn_reduced_hamiltonian(
    relative_position: Array,
    relative_momentum: Array,
    masses: Array,
    G: float,
    c: float,
) -> float:
    """Reduced ADM two-body Hamiltonian through conservative 2PN order.

    This implements Eqs. (6.4)-(6.9) of Schäfer & Jaranowski (2018),
    Hamiltonian formulation of general relativity and post-Newtonian dynamics
    of compact binaries, in reduced center-of-mass variables.
    """
    validate_pn_inputs(
        np.asarray([[0.0, 0.0, 0.0], relative_position], dtype=np.float64),
        np.asarray([[0.0, 0.0, 0.0], relative_momentum], dtype=np.float64),
        masses,
        "two-body",
    )
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 2PN dynamics")

    total_mass = float(np.sum(masses))
    reduced_mass = float(np.prod(masses) / total_mass)
    nu = reduced_mass / total_mass

    r_vec = np.asarray(relative_position, dtype=np.float64) / (G * total_mass)
    p_vec = np.asarray(relative_momentum, dtype=np.float64) / reduced_mass
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("2PN Hamiltonian is undefined for zero separation")
    n = r_vec / r
    p2 = float(np.dot(p_vec, p_vec))
    pr = float(np.dot(n, p_vec))

    h_n = 0.5 * p2 - 1.0 / r
    h_1pn = (
        0.125 * (3.0 * nu - 1.0) * p2 * p2
        - 0.5 * ((3.0 + nu) * p2 + nu * pr * pr) / r
        + 0.5 / (r * r)
    )
    h_2pn = (
        0.0625 * (1.0 - 5.0 * nu + 5.0 * nu * nu) * p2 * p2 * p2
        + 0.125
        * (
            (5.0 - 20.0 * nu - 3.0 * nu * nu) * p2 * p2
            - 2.0 * nu * nu * pr * pr * p2
            - 3.0 * nu * nu * pr**4
        )
        / r
        + 0.5 * ((5.0 + 8.0 * nu) * p2 + 3.0 * nu * pr * pr) / (r * r)
        - 0.25 * (1.0 + 3.0 * nu) / (r**3)
    )
    return float(h_n + h_1pn / (c * c) + h_2pn / (c * c * c * c))


def two_body_adm_2pn_rhs(
    relative_position: Array,
    relative_momentum: Array,
    masses: Array,
    G: float,
    c: float,
) -> tuple[Array, Array]:
    """Hamilton equations for the reduced ADM two-body Hamiltonian through 2PN."""
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 2PN dynamics")

    total_mass = float(np.sum(masses))
    reduced_mass = float(np.prod(masses) / total_mass)
    nu = reduced_mass / total_mass

    r_vec = np.asarray(relative_position, dtype=np.float64) / (G * total_mass)
    p_vec = np.asarray(relative_momentum, dtype=np.float64) / reduced_mass
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("2PN Hamiltonian is undefined for zero separation")
    n = r_vec / r
    p2 = float(np.dot(p_vec, p_vec))
    pr = float(np.dot(n, p_vec))
    p_perp = p_vec - pr * n

    d_hn_dp = p_vec
    d_h1_dp = 0.5 * (3.0 * nu - 1.0) * p2 * p_vec - ((3.0 + nu) * p_vec + nu * pr * n) / r
    coeff = 5.0 - 20.0 * nu - 3.0 * nu * nu
    d_h2_dp = (
        0.375 * (1.0 - 5.0 * nu + 5.0 * nu * nu) * p2 * p2 * p_vec
        + 0.5 * (coeff * p2 * p_vec - nu * nu * pr * p2 * n - nu * nu * pr * pr * p_vec - 3.0 * nu * nu * pr**3 * n) / r
        + ((5.0 + 8.0 * nu) * p_vec + 3.0 * nu * pr * n) / (r * r)
    )
    d_r_dt = d_hn_dp + d_h1_dp / (c * c) + d_h2_dp / (c * c * c * c)

    c1 = (3.0 + nu) * p2 + nu * pr * pr
    grad_h1 = 0.5 * c1 * n / (r * r) - nu * pr * p_perp / (r * r) - n / (r**3)

    d_term = coeff * p2 * p2 - 2.0 * nu * nu * pr * pr * p2 - 3.0 * nu * nu * pr**4
    grad_h2 = (
        -0.5 * nu * nu * pr * (p2 + 3.0 * pr * pr) * p_perp / (r * r)
        - d_term * n / (8.0 * r * r)
        + 3.0 * nu * pr * p_perp / (r**3)
        - ((5.0 + 8.0 * nu) * p2 + 3.0 * nu * pr * pr) * n / (r**3)
        + 0.75 * (1.0 + 3.0 * nu) * n / (r**4)
    )
    d_p_dt_reduced = -(n / (r * r) + grad_h1 / (c * c) + grad_h2 / (c * c * c * c))

    d_relative_position_dt = d_r_dt
    d_relative_momentum_dt = reduced_mass * d_p_dt_reduced / (G * total_mass)
    return d_relative_position_dt, d_relative_momentum_dt


def solve_two_body_adm_2pn_momentum_from_velocity(
    relative_position: Array,
    relative_velocity: Array,
    masses: Array,
    G: float,
    c: float,
    *,
    tolerance: float = 1e-15,
    max_iterations: int = 8,
) -> Array:
    """Solve for the ADM canonical relative momentum matching a target velocity.

    The two-body ADM Hamiltonian uses canonical momenta rather than the
    Newtonian relation p = mu v. This routine inverts dr/dt = dH/dp with a
    small Newton solve so the canonical initial data matches the requested
    physical relative velocity.
    """
    validate_pn_inputs(
        np.asarray([[0.0, 0.0, 0.0], relative_position], dtype=np.float64),
        np.asarray([[0.0, 0.0, 0.0], relative_velocity], dtype=np.float64),
        masses,
        "two-body",
    )
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 2PN dynamics")

    target_velocity = np.asarray(relative_velocity, dtype=np.float64)
    reduced_mass = float(np.prod(masses) / np.sum(masses))
    momentum = reduced_mass * target_velocity.copy()

    velocity_scale = max(1.0, float(np.linalg.norm(target_velocity)))
    momentum_scale = max(1.0, float(np.linalg.norm(momentum)))
    for _ in range(max_iterations):
        current_velocity, _ = two_body_adm_2pn_rhs(relative_position, momentum, masses, G, c)
        residual = current_velocity - target_velocity
        if float(np.linalg.norm(residual)) <= tolerance * velocity_scale:
            return momentum

        jacobian = np.empty((3, 3), dtype=np.float64)
        for axis in range(3):
            step = 1e-8 * max(1.0, abs(momentum[axis]))
            plus = momentum.copy()
            minus = momentum.copy()
            plus[axis] += step
            minus[axis] -= step
            velocity_plus, _ = two_body_adm_2pn_rhs(relative_position, plus, masses, G, c)
            velocity_minus, _ = two_body_adm_2pn_rhs(relative_position, minus, masses, G, c)
            jacobian[:, axis] = (velocity_plus - velocity_minus) / (2.0 * step)

        try:
            correction = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Failed to invert the ADM 2PN velocity-momentum Jacobian") from exc
        momentum -= correction
        if float(np.linalg.norm(correction)) <= tolerance * momentum_scale:
            return momentum

    final_velocity, _ = two_body_adm_2pn_rhs(relative_position, momentum, masses, G, c)
    final_residual = float(np.linalg.norm(final_velocity - target_velocity))
    raise RuntimeError(
        "ADM 2PN momentum solve did not converge; "
        f"final velocity mismatch was {final_residual:.3e}"
    )


def total_energy_2pn_two_body_from_momenta(
    positions: Array,
    momenta: Array,
    masses: Array,
    G: float,
    c: float,
) -> float:
    """Conservative two-body ADM 2PN binding energy from canonical momenta."""
    validate_pn_inputs(positions, momenta, masses, "two-body")
    relative_position = positions[1] - positions[0]
    relative_momentum = 0.5 * (momenta[0] - momenta[1])
    reduced_h = two_body_adm_2pn_reduced_hamiltonian(relative_position, relative_momentum, masses, G, c)
    reduced_mass = float(np.prod(masses) / float(np.sum(masses)))
    return float(reduced_mass * reduced_h)


def total_energy_2pn_two_body(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    c: float,
) -> float:
    """Conservative two-body ADM 2PN binding energy in the center-of-mass frame."""
    validate_pn_inputs(positions, velocities, masses, "two-body")
    relative_position = positions[1] - positions[0]
    total_mass = float(np.sum(masses))
    reduced_mass = float(np.prod(masses) / total_mass)
    relative_momentum = reduced_mass * (velocities[1] - velocities[0])
    reduced_h = two_body_adm_2pn_reduced_hamiltonian(relative_position, relative_momentum, masses, G, c)
    return float(reduced_mass * reduced_h)


def three_body_adm_hamiltonian_through_2pn(
    positions: Array,
    momenta: Array,
    masses: Array,
    G: float,
    c: float,
) -> float:
    """ADM three-body Hamiltonian through conservative 2PN order.

    The 2PN sector follows the explicit appendix formula in Lousto & Nakano
    (2008), which corrects typos in Schäfer (1987). Units are physical with
    explicit G and c factors restored.
    """
    validate_pn_inputs(positions, momenta, masses, "eih")
    if positions.shape[0] != 3:
        raise ValueError("The current many-body ADM 2PN Hamiltonian is implemented for exactly three bodies")
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 2PN dynamics")

    x = np.asarray(positions, dtype=np.float64)
    p = np.asarray(momenta, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    n_bodies = 3

    def r_ab(a: int, b: int) -> float:
        return float(np.linalg.norm(x[a] - x[b]))

    def n_ab(a: int, b: int) -> Array:
        r = r_ab(a, b)
        if r == 0.0:
            raise ValueError("Hamiltonian is undefined for zero separation")
        return (x[a] - x[b]) / r

    def p2(a: int) -> float:
        return float(np.dot(p[a], p[a]))

    h_n = 0.5 * sum(p2(a) / m[a] for a in range(n_bodies))
    h_n -= 0.5 * G * sum(m[a] * m[b] / r_ab(a, b) for a in range(n_bodies) for b in range(n_bodies) if b != a)

    h_1pn = -0.125 * sum(m[a] * (p2(a) / (m[a] * m[a])) ** 2 for a in range(n_bodies))
    pair_sum = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            nab = n_ab(a, b)
            pair_sum += (m[a] * m[b] / r_ab(a, b)) * (
                6.0 * p2(a) / (m[a] * m[a])
                - 7.0 * float(np.dot(p[a], p[b])) / (m[a] * m[b])
                - float(np.dot(nab, p[a])) * float(np.dot(nab, p[b])) / (m[a] * m[b])
            )
    h_1pn -= 0.25 * G * pair_sum
    triple_sum = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            for c_idx in range(n_bodies):
                if c_idx == a:
                    continue
                triple_sum += m[a] * m[b] * m[c_idx] / (r_ab(a, b) * r_ab(a, c_idx))
    h_1pn += 0.5 * G * G * triple_sum

    h_2pn = 0.0625 * sum(m[a] * (p2(a) / (m[a] * m[a])) ** 3 for a in range(n_bodies))
    term2 = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            nab = n_ab(a, b)
            na_pa = float(np.dot(nab, p[a]))
            na_pb = float(np.dot(nab, p[b]))
            pa_pb = float(np.dot(p[a], p[b]))
            term2 += (m[a] * m[b] / r_ab(a, b)) * (
                10.0 * (p2(a) / (m[a] * m[a])) ** 2
                - 11.0 * p2(a) * p2(b) / (m[a] * m[a] * m[b] * m[b])
                - 2.0 * (pa_pb * pa_pb) / (m[a] * m[a] * m[b] * m[b])
                + 10.0 * p2(a) * (na_pb * na_pb) / (m[a] * m[a] * m[b] * m[b])
                - 12.0 * pa_pb * na_pa * na_pb / (m[a] * m[a] * m[b] * m[b])
                - 3.0 * (na_pa * na_pa) * (na_pb * na_pb) / (m[a] * m[a] * m[b] * m[b])
            )
    h_2pn += 0.0625 * G * term2

    term3 = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            for c_idx in range(n_bodies):
                if c_idx == a:
                    continue
                nab = n_ab(a, b)
                nac = n_ab(a, c_idx)
                term3 += (m[a] * m[b] * m[c_idx] / (r_ab(a, b) * r_ab(a, c_idx))) * (
                    18.0 * p2(a) / (m[a] * m[a])
                    + 14.0 * p2(b) / (m[b] * m[b])
                    - 2.0 * (float(np.dot(nab, p[b])) ** 2) / (m[b] * m[b])
                    - 50.0 * float(np.dot(p[a], p[b])) / (m[a] * m[b])
                    + 17.0 * float(np.dot(p[b], p[c_idx])) / (m[b] * m[c_idx])
                    - 14.0 * float(np.dot(nab, p[a])) * float(np.dot(nab, p[b])) / (m[a] * m[b])
                    + 14.0 * float(np.dot(nab, p[b])) * float(np.dot(nab, p[c_idx])) / (m[b] * m[c_idx])
                    + float(np.dot(nab, nac))
                    * float(np.dot(nab, p[b]))
                    * float(np.dot(nac, p[c_idx]))
                    / (m[b] * m[c_idx])
                )
    h_2pn += 0.125 * G * G * term3

    term4 = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            for c_idx in range(n_bodies):
                if c_idx == a:
                    continue
                nab = n_ab(a, b)
                nac = n_ab(a, c_idx)
                term4 += (m[a] * m[b] * m[c_idx] / (r_ab(a, b) ** 2)) * (
                    2.0 * float(np.dot(nab, p[a])) * float(np.dot(nac, p[c_idx])) / (m[a] * m[c_idx])
                    + 2.0 * float(np.dot(nab, p[b])) * float(np.dot(nac, p[c_idx])) / (m[a] * m[c_idx])
                    + 5.0 * float(np.dot(nab, nac)) * p2(c_idx) / (m[c_idx] * m[c_idx])
                    - float(np.dot(nab, nac)) * (float(np.dot(nac, p[c_idx])) ** 2) / (m[c_idx] * m[c_idx])
                    - 14.0 * float(np.dot(nab, p[c_idx])) * float(np.dot(nac, p[c_idx])) / (m[c_idx] * m[c_idx])
                )
    h_2pn += 0.125 * G * G * term4

    term5 = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            term5 += (m[a] * m[a] * m[b] / (r_ab(a, b) ** 2)) * (
                p2(a) / (m[a] * m[a])
                + p2(b) / (m[b] * m[b])
                - 2.0 * float(np.dot(p[a], p[b])) / (m[a] * m[b])
            )
    h_2pn += 0.25 * G * G * term5

    term6 = 0.0
    term7 = 0.0
    term8 = 0.0
    term9 = 0.0
    term10 = 0.0
    term11 = 0.0
    term12 = 0.0
    term13 = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            for c_idx in range(n_bodies):
                if c_idx == a or c_idx == b:
                    continue
                rab = r_ab(a, b)
                rbc = r_ab(b, c_idx)
                rca = r_ab(c_idx, a)
                denom = rab + rbc + rca
                nab = n_ab(a, b)
                nac = n_ab(a, c_idx)
                ncb = n_ab(c_idx, b)
                vec_i = nab + nac
                vec_j = nab + ncb
                tensor = (
                    8.0 * np.outer(p[a], p[c_idx]) / (m[a] * m[c_idx])
                    - 16.0 * np.outer(p[c_idx], p[a]) / (m[a] * m[c_idx])
                    + 3.0 * np.outer(p[a], p[b]) / (m[a] * m[b])
                    + 4.0 * np.outer(p[c_idx], p[c_idx]) / (m[c_idx] * m[c_idx])
                    + np.outer(p[a], p[a]) / (m[a] * m[a])
                )
                term6 += (m[a] * m[b] * m[c_idx] / (denom * denom)) * float(vec_i @ tensor @ vec_j)
                term7 += (m[a] * m[b] * m[c_idx] / (denom * rab)) * (
                    8.0
                    * (
                        float(np.dot(p[a], p[c_idx]))
                        - float(np.dot(nab, p[a])) * float(np.dot(nab, p[c_idx]))
                    )
                    / (m[a] * m[c_idx])
                    - 3.0
                    * (
                        float(np.dot(p[a], p[b]))
                        - float(np.dot(nab, p[a])) * float(np.dot(nab, p[b]))
                    )
                    / (m[a] * m[b])
                    - 4.0 * (p2(c_idx) - float(np.dot(nab, p[c_idx])) ** 2) / (m[c_idx] * m[c_idx])
                    - (p2(a) - float(np.dot(nab, p[a])) ** 2) / (m[a] * m[a])
                )
                term8 += m[a] * m[a] * m[b] * m[c_idx] / (rab * rab * rbc)
                term9 += m[a] * m[b] * m[c_idx] * m[c_idx] / (rab * rca * rca)
                term10 += m[a] * m[a] * m[b] * m[c_idx] / (rab * rab * rca)
                term11 += m[a] * m[a] * m[b] * m[c_idx] / (rab * rca * rbc)
                poly = (
                    18.0 * rab * rab * rca * rca
                    - 60.0 * rab * rab * rbc * rbc
                    - 24.0 * rab * rab * rca * (rab + rbc)
                    + 60.0 * rab * rca * rbc * rbc
                    + 56.0 * rab**3 * rbc
                    - 72.0 * rab * rbc**3
                    + 35.0 * rbc**4
                    + 6.0 * rab**4
                )
                term13 += (m[a] * m[a] * m[b] * m[c_idx] * poly) / (rab**3 * rca**3 * rbc)
    h_2pn += 0.5 * G * G * term6
    h_2pn += 0.5 * G * G * term7
    h_2pn -= 0.5 * G * G * G * term8
    h_2pn -= 0.25 * G * G * G * term9

    term10_pair = 0.0
    term12_pair = 0.0
    for a in range(n_bodies):
        for b in range(n_bodies):
            if b == a:
                continue
            term10_pair += m[a] ** 3 * m[b] / (r_ab(a, b) ** 3)
            term12_pair += m[a] * m[a] * m[b] * m[b] / (r_ab(a, b) ** 3)
    h_2pn += 0.5 * G * G * G * term10_pair
    h_2pn -= 0.75 * G * G * G * term10
    h_2pn -= 0.375 * G * G * G * term11
    h_2pn += 0.375 * G * G * G * term12_pair
    h_2pn -= (1.0 / 64.0) * G * G * G * term13
    h_2pn -= 0.25 * G * G * G * term12_pair

    return float(h_n + h_1pn / (c * c) + h_2pn / (c * c * c * c))


def eih_energy_components(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    c: float,
) -> tuple[float, float, float]:
    """Return a diagnostic bookkeeping split for the EIH 1PN conserved energy.

    The conserved total energy is the physically meaningful quantity. The
    returned `(K, U, E)` split is a practical diagnostic partition derived from
    the 1PN N-body Lagrangian: `K` contains the pure one-body kinetic terms,
    while `U` absorbs all interaction-dependent contributions, including the
    velocity-dependent pair terms. This split is not unique in the same sense as
    the Newtonian `K/U` decomposition.
    """
    validate_pn_inputs(positions, velocities, masses, "eih")
    if c <= 0.0:
        raise ValueError("The effective speed of light c must be positive for 1PN energy")

    _, _, inv_dist, unit_ba = _pairwise_geometry(positions, 0.0)
    v2 = np.sum(velocities * velocities, axis=1)
    masses_outer = masses[:, None] * masses[None, :]
    pair_mass_over_r = masses_outer * inv_dist
    n_ab = -unit_ba
    n_ab_dot_va = np.sum(n_ab * velocities[:, None, :], axis=2)
    n_ab_dot_vb = np.sum(n_ab * velocities[None, :, :], axis=2)
    vv = velocities @ velocities.T
    mass_sum_over_r = np.sum(masses[None, :] * inv_dist, axis=1)

    kinetic = 0.5 * np.sum(masses * v2) + 0.375 * np.sum(masses * v2 * v2) / (c * c)
    interaction = -0.5 * G * np.sum(pair_mass_over_r)
    interaction += (
        1.5 * G * np.sum(masses * v2 * mass_sum_over_r)
        - 0.25 * G * np.sum(pair_mass_over_r * (7.0 * vv + n_ab_dot_va * n_ab_dot_vb))
        + 0.5 * G * G * np.sum(masses * mass_sum_over_r * mass_sum_over_r)
    ) / (c * c)
    total = kinetic + interaction
    return float(kinetic), float(interaction), float(total)


def combined_2pn_acceleration(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float,
    eps: float,
    c: float,
    scope: PNScope = "two-body",
    primary_index: int = 0,
) -> Array:
    validate_2pn_inputs(positions, velocities, masses, scope, primary_index=primary_index)
    del positions, velocities, masses, G, eps, c, primary_index
    raise canonical_2pn_not_implemented(scope)
