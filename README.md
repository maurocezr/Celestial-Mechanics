# Celestial Mechanics

A compact NumPy-based orbital dynamics sandbox for Newtonian N-body simulations and selected
post-Newtonian experiments. It includes Solar System and compact-binary presets, trajectory and
energy diagnostics, periapsis/precession analysis tools, and a simple CLI.

> Phase 1 — Foundations: focus on classical gravity (N-body) in Python with NumPy; later you can port
> the kernels to C++/Rust or run on GPU (PyTorch/CUDA).

---

## Features

- Physics: Newtonian gravity with Plummer-like softening (avoids singular forces)
- Integrators:
  - Leapfrog (KDK / velocity-Verlet) — symplectic, ideal for long-term energy behavior
  - RK4 — classic 4th-order, accurate per-step but non-symplectic
- Relativistic bridge support:
  - `1PN` force model for exact two-body, central-body, and EIH experiments
  - `2PN` canonical ADM two-body backend plus a three-body milestone Hamiltonian path
  - `2.5PN` complete two-body harmonic model through Newtonian + `1PN` + `2PN` + `2.5PN`
  - Mercury-style preset for perihelion precession studies
  - compact-binary toy preset for stronger relativistic orbital deviations
  - inner-solar-system toy preset for central-body `1PN` experiments
  - PN logic isolated in `relativity.py`
- Presets:
  - two-body (Sun-Earth; units: AU, yr, Msun; G = 4*pi^2)
  - solar-system-horizons (Sun + planets + Pluto from JPL/Horizons; units: AU, yr, Msun)
  - mercury-relativistic (Sun-Mercury; units: AU, yr, Msun; intended for `1PN` runs)
  - binary-pulsar-toy (equal-mass compact binary; units: AU, yr, Msun; intended for `1PN` comparisons)
  - psr-b1913+16 (Hulse-Taylor binary; units: AU, yr, Msun; intended for `eih`/compact-binary validation)
  - psr-b1534+12 (double-neutron-star binary; units: AU, yr, Msun; intended for `eih`/compact-binary validation)
  - psr-j0737-3039ab (Double Pulsar; units: AU, yr, Msun; intended for `eih`/compact-binary validation)
  - psr-j1757-1854 (high-eccentricity double-neutron-star binary; units: AU, yr, Msun; intended for `eih`/compact-binary validation)
  - inner-solar-system-toy (Sun + Mercury + Venus + Earth; units: AU, yr, Msun; intended for `central-body` `1PN`)
  - three-body-figure8 (canonical choreography; G = 1)
  - star-earth-jupiter (3-body with realistic mass ratios; G = 4*pi^2)
  - random (quick generator for large-N experiments; G = 1)
  - ephemeris (loads state vectors from json/csv/npz files or JPL/Horizons)
- Diagnostics: kinetic K, potential U, total energy E, relative drift dE/E0
- Analysis helpers:
  - periapsis sampling with quadratic interpolation
  - frequency-based precession estimation from radial and azimuthal motion
  - orbit-comparison and planetary-precession validation CLIs
- Center-of-mass correction: removes COM translation from ICs by default
- CLI: pick integrator, time step, softening, steps/years, sampling, and output


## Current Status

- Newtonian N-body is the stable baseline workflow.
- `1PN` is the most mature relativistic path:
  - exact comparable-mass two-body mode
  - central-body approximation for Solar System experiments
  - full EIH mode for compact-binary and many-body `1PN` validation
- `2PN` currently means:
  - exact canonical ADM two-body evolution in the center-of-mass frame
  - explicit ADM three-body Hamiltonian support via numerical Hamilton equations
- `2.5PN` currently means:
  - a force-based two-body harmonic model through Newtonian + `1PN` + `2PN` + `2.5PN`
  - useful for inspiral experiments, but still less mature than the `1PN` validation tooling
- `fit_double_pulsar_2pn.py` now builds a direct canonical `2PN` Double Pulsar initial state from
  the published orbital parameters and the Hamiltonian velocity-to-momentum inversion.


## Installation

Requirements:

- Python 3.9+
- NumPy (required)
- Matplotlib (optional for plots)

```bash
pip install numpy matplotlib
```

Clone/download the repo (or just the script) and run from the project root:

```bash
python nbody.py --help
```


## Quick Start

Two-body (Sun-Earth) with Leapfrog, 1 year, small dt:
```bash
python nbody.py --preset two-body --integrator leapfrog --years 1 --dt 0.001 --plot
```

Three-body figure-8 (dimensionless) with RK4:
```bash
python nbody.py --preset three-body-figure8 --integrator rk4 --steps 10000 --dt 0.001 --plot
```

Mercury-style relativistic bridge run with `1PN` and RK4:
```bash
python nbody.py --preset mercury-relativistic --gravity-model 1pn --pn-scope two-body --integrator rk4 --c 63239.7263 --years 1 --dt 0.0005 --plot
```

Compact-binary toy run with `1PN` and RK4:
```bash
python nbody.py --preset binary-pulsar-toy --gravity-model 1pn --pn-scope two-body --integrator rk4 --c 63239.7263 --years 0.2 --dt 0.00001 --plot
```

PSR B1913+16 run with `1PN` EIH and RK4:
```bash
python nbody.py --preset psr-b1913+16 --gravity-model 1pn --pn-scope eih --integrator rk4 --c 63239.7263 --years 0.2 --dt 0.000001 --no-plot
```

PSR B1534+12 run with `1PN` EIH and RK4:
```bash
python nbody.py --preset psr-b1534+12 --gravity-model 1pn --pn-scope eih --integrator rk4 --c 63239.7263 --years 0.5 --dt 0.0000015 --no-plot
```

PSR J0737-3039A/B run with `1PN` EIH and RK4:
```bash
python nbody.py --preset psr-j0737-3039ab --gravity-model 1pn --pn-scope eih --integrator rk4 --c 63239.7263 --years 0.02 --dt 0.000001 --eps 1e-9 --save-every 1 --no-plot
```

PSR J1757-1854 run with `1PN` EIH and RK4:
```bash
python nbody.py --preset psr-j1757-1854 --gravity-model 1pn --pn-scope eih --integrator rk4 --c 63239.7263 --years 0.03 --dt 0.000001 --eps 1e-9 --save-every 1 --no-plot
```

Mercury run with canonical ADM `2PN` two-body dynamics:
```bash
python nbody.py --preset mercury-relativistic --gravity-model 2pn --pn-scope two-body --integrator rk4 --c 63239.7263 --years 0.05 --dt 0.0002 --no-plot
```

Compact-binary toy run with the complete two-body harmonic `2.5PN` force model:
```bash
python nbody.py --preset binary-pulsar-toy --gravity-model 2.5pn --pn-scope two-body --integrator rk4 --c 63239.7263 --years 0.05 --dt 0.00001 --no-plot
```

Three-body run with ADM `2PN` Hamiltonian dynamics (milestone 1 many-body path):
```bash
python nbody.py --preset star-earth-jupiter --gravity-model 2pn --pn-scope eih --integrator rk4 --c 63239.7263 --years 0.01 --dt 0.001 --no-plot
```

Inner Solar System toy run with `central-body` `1PN` sourced by the Sun:
```bash
python nbody.py --preset inner-solar-system-toy --gravity-model 1pn --pn-scope central-body --pn-primary-index 0 --integrator rk4 --c 63239.7263 --years 5 --dt 0.0002 --plot
```

Star-Earth-Jupiter (20 years) with Leapfrog:
```bash
python nbody.py --preset star-earth-jupiter --integrator leapfrog --years 20 --dt 0.001 --plot
```

Random N-body (e.g., 1000 particles) with Leapfrog:
```bash
python nbody.py --preset random --n 1000 --integrator leapfrog --years 2 --dt 0.002 --plot
```

Ephemeris-driven initial conditions from a JSON file:
```bash
python nbody.py --preset ephemeris --ephemeris-file examples/earth_moon.json --ephemeris-g 39.47841760435743 --years 0.1 --dt 0.0005 --plot
```

Ephemeris-driven initial conditions from JPL/Horizons:
```bash
python nbody.py --preset ephemeris --ephemeris-source horizons --horizons-target 10 --horizons-target 399 --horizons-epoch "2026-01-01 00:00" --years 1 --dt 0.001 --plot
```

Full Solar System (Sun + planets + Pluto) from JPL/Horizons:
```bash
python nbody.py --preset solar-system-horizons --horizons-epoch "2026-01-01 00:00" --years 5 --dt 0.001 --plot --plot-mode 3d
```

Save results for post-processing:
```bash
python nbody.py --preset random --n 500 --integrator leapfrog --years 1 --dt 0.002 --save run_random_500.npz
```

The saved .npz contains: times, traj, vel, K, U, energy, masses, G, eps, dt, preset, integrator, note, and when applicable `gravity_model`, `pn_scope`, `c`.

Build a direct canonical `2PN` Double Pulsar initial state:
```bash
python fit_double_pulsar_2pn.py --save-state psr_j0737_3039ab_2pn_direct_state.npz
```

Render a video from a saved result without adding rendering logic to `nbody.py`:
```bash
python render_video.py run_random_500.npz run_random_500.mp4 --fps 30 --stride 2
```

Or write a GIF if `Pillow` is available:
```bash
python render_video.py run_random_500.npz run_random_500.gif --fps 20 --tail 100
```

Measure perihelion precession from saved trajectories:
```bash
python analyze_precession.py mercury_1pn.npz --compare mercury_newtonian.npz
```

Compare orbital changes between two saved runs:
```bash
python compare_orbits.py mercury_newtonian.npz mercury_1pn.npz
```

Validate baseline-subtracted planetary precession against bundled reference values:
```bash
python validate_planetary_precession.py solar_system_newtonian.npz solar_system_1pn.npz --primary-index 0
```

Report orbital drift plus precession validation for the whole Solar System in one pass:
```bash
python report_solar_system_validation.py solar_system_newtonian.npz solar_system_1pn.npz --primary-index 0 --profile inner-planets
```


## Command-Line Reference

```
--preset {two-body, three-body-figure8, star-earth-jupiter, random, ephemeris, solar-system-horizons}
                  # plus mercury-relativistic, binary-pulsar-toy, psr-b1913+16, psr-b1534+12, psr-j0737-3039ab, psr-j1757-1854, inner-solar-system-toy
--integrator {leapfrog, rk4}
--steps INT            # total integration steps (overrides --years if given)
--years FLOAT          # convenience for AU/yr presets (steps = years/dt)
--dt FLOAT             # time step
--save-every INT       # save every k steps (reduces memory)
--eps FLOAT            # gravitational softening length (position units)
--gravity-model {newtonian,1pn,2pn,2.5pn}
--c FLOAT              # effective speed of light in simulation units for PN runs
--pn-scope {none,two-body,central-body,eih}
--pn-primary-index INT # primary body for central-body PN runs
--n INT                # number of particles for --preset random
--seed INT
--mass-spread FLOAT    # fractional mass spread for random ICs (0 = equal masses)
--ephemeris-source {file,horizons}
--ephemeris-file PATH  # required for --preset ephemeris
--ephemeris-format {auto,json,csv,npz}
--ephemeris-g FLOAT    # overrides G in the ephemeris file
--ephemeris-note TEXT  # overrides metadata note for the ephemeris file
--horizons-target TEXT # repeat for each Horizons COMMAND target
--horizons-epoch TEXT  # required for --ephemeris-source horizons
--horizons-center TEXT # Horizons center code, default 500@0
--horizons-ref-plane {ecliptic,frame}
--horizons-mass KEY=M  # optional mass override in solar masses
--horizons-ssl-insecure # disable TLS verification for trusted proxy environments
--ephemeris-no-com     # keep the supplied frame instead of shifting to COM
--plot-mode {2d,3d}    # trajectory visualization mode
--plot / --no-plot
--save PATH.npz
```

`render_video.py` accepts:

```
INPUT.npz OUTPUT.{mp4,gif}
--plot-mode {2d,3d}
--fps INT
--stride INT         # render every k-th saved frame
--dpi INT
--tail INT           # show only the last N frames of each trajectory; 0 = full path
--marker-size FLOAT
--line-width FLOAT
--title TEXT
```

`analyze_precession.py` accepts:

```
INPUT.npz
--primary-index INT
--secondary-index INT
--compare BASELINE.npz
```

`compare_orbits.py` accepts:

```
REFERENCE.npz CANDIDATE.npz
--primary-index INT
--secondary-index INT
```

`validate_planetary_precession.py` accepts:

```
REFERENCE.npz CANDIDATE.npz
--primary-index INT
```

`report_solar_system_validation.py` accepts:

```
REFERENCE.npz CANDIDATE.npz
--primary-index INT
--profile {inner-planets,through-jupiter,outer-planets}
```


## Units and Constants

- Astronomical presets (two-body, star-earth-jupiter):
  - Units: AU, year, Msun
  - Gravitational constant: G = 4*pi^2 so that a circular orbit at 1 AU around 1 Msun
    has speed ~ 2*pi AU/yr and period 1 year
- Dimensionless presets (three-body-figure8, random): G = 1


## Initial Conditions (ICs)

- ICs are obtained from presets. After construction, the code automatically shifts
  to the center-of-mass frame, ensuring sum(m_i * r_i) = 0 and sum(m_i * v_i) = 0.
- Softening eps is applied both in the acceleration and potential to avoid singularities and
  improve stability during close encounters.

### Ephemeris ICs

Use `--preset ephemeris` to load state vectors from either a local file or the JPL/Horizons API at a chosen epoch.
Use `--preset solar-system-horizons` as a shorthand for Sun + planets + Pluto from the JPL/Horizons API.

File-backed ephemerides:

- `json`: top-level object with optional `G`, `note`, `epoch`, and a `bodies` list
- `csv`: one body per row with columns `mass,x,y,z,vx,vy,vz` and optional `name`
- `npz`: arrays `positions`, `velocities`, `masses` and optional `G`, `note`, `epoch`, `names`

JSON example:

```json
{
  "G": 39.47841760435743,
  "note": "Units: AU, yr, Msun",
  "epoch": "2026-01-01T00:00:00",
  "bodies": [
    {
      "name": "Sun",
      "mass": 1.0,
      "position": [0.0, 0.0, 0.0],
      "velocity": [0.0, 0.0, 0.0]
    },
    {
      "name": "Earth",
      "mass": 3.00348961491547e-6,
      "position": [1.0, 0.0, 0.0],
      "velocity": [0.0, 6.283185307179586, 0.0]
    }
  ]
}
```

By default the loader shifts ephemeris ICs to the center-of-mass frame, matching the behavior
of the built-in presets. Use `--ephemeris-no-com` if your ephemeris is already expressed in the
frame you want to integrate in.

Horizons-backed ephemerides:

- Set `--ephemeris-source horizons`
- Provide one or more `--horizons-target` values using Horizons `COMMAND` identifiers such as `10` (Sun), `399` (Earth), `499` (Mars), etc.
- Provide a single `--horizons-epoch` value; the implementation fetches one state vector per target using the Horizons `VECTORS` API
- States are converted to `AU`, `yr`, `Msun` units, so the default gravitational constant becomes `G = 4*pi^2`
- Built-in masses are included for Sun, Moon, and the major planets. For any other target, pass `--horizons-mass TARGET=MASS_IN_MSUN`
- If your environment terminates TLS with a trusted-but-local proxy certificate, `--horizons-ssl-insecure` disables certificate verification for the request

Common Horizons target codes:

- `10`: Sun
- `199`: Mercury
- `299`: Venus
- `399`: Earth
- `301`: Moon
- `499`: Mars
- `599`: Jupiter
- `699`: Saturn
- `799`: Uranus
- `899`: Neptune
- `999`: Pluto

Example full Solar System command:

```bash
python nbody.py --preset ephemeris --ephemeris-source horizons --horizons-target 10 --horizons-target 199 --horizons-target 299 --horizons-target 399 --horizons-target 499 --horizons-target 599 --horizons-target 699 --horizons-target 799 --horizons-target 899 --horizons-target 999 --horizons-epoch "2026-01-01 00:00" --years 5 --dt 0.001 --plot
```

Example with an explicit mass override:

```bash
python nbody.py --preset ephemeris --ephemeris-source horizons --horizons-target 10 --horizons-target 301 --horizons-epoch "2026-01-01 00:00" --horizons-mass 301=3.694303349765111e-8 --years 0.1 --dt 0.0005 --plot
```


## Integrators and Stability

- Leapfrog (Kick-Drift-Kick) is symplectic -> excellent long-term energy behavior,
  typically showing small bounded oscillations in energy.
- RK4 provides high accuracy per step but is not symplectic, so it can exhibit gradual energy drift over long integrations.
- `1PN` runs currently require `RK4` because the relativistic correction depends on both position and velocity.
- If you see instability or large dE/E0:
  - Reduce --dt
  - Increase --eps slightly
  - Prefer Leapfrog for long-term runs


## Relativistic Corrections

- `--gravity-model newtonian` is the default and preserves the original solver behavior.
- `--gravity-model 1pn` enables the first Post-Newtonian bridge step for weak-field two-body and central-body experiments.
- The current `1PN` implementation supports:
  - an exact comparable-mass two-body harmonic-coordinate mode
  - a central-body approximation in which one designated primary sources the relativistic correction
  - a full harmonic-coordinate Einstein-Infeld-Hoffmann (`eih`) N-body mode
- `--gravity-model 2pn` now supports an exact canonical ADM two-body backend in the center-of-mass frame.
- `--gravity-model 2.5pn` now evaluates the complete scoped two-body harmonic force model through Newtonian + `1PN` + `2PN` + `2.5PN`.
- Milestone 1 of the many-body path is also implemented: the explicit ADM three-body `2PN` Hamiltonian is available through the canonical backend using numerical Hamilton equations.
- Broader many-body `2PN` (`N > 3`) is still not implemented, but the runtime is isolated from the validated force-based regimes and includes a tested canonical phase-space RK4 scaffold.
- The standalone two-body `1PN` force path and the ADM canonical `2PN` backend remain separate implementations; the `2.5PN` branch does not replace either of them.
- The current `2.5PN` path is limited to `--pn-scope two-body`, still uses a bookkeeping `1PN` energy diagnostic because radiation reaction is dissipative, and should be treated as a controlled compact-binary experiment rather than a precision timing model.
- Use `--pn-scope two-body` for exact two-body experiments, `--pn-scope central-body --pn-primary-index 0` for primary-sourced approximations, and `--pn-scope eih` for full `1PN` N-body experiments.
- The recommended starting preset is `--preset mercury-relativistic`.
- The right-hand plot panel shows Newtonian `dE/E0` for Newtonian runs and a two-body `1PN` total-energy drift for `1PN` runs.
- In `eih` mode, the right-hand plot panel shows the conserved EIH total-energy drift and a bookkeeping `K/U` split derived from the 1PN N-body Lagrangian. Only the total EIH energy is physically unambiguous; the `K/U` partition is diagnostic rather than unique.
- `binary-pulsar-toy` is a demonstration preset for stronger two-body relativistic deviations. It is not a full compact-object timing model and should be treated as a controlled toy case.
- `inner-solar-system-toy` is a demonstration preset for Newtonian planet-planet perturbations plus Sun-sourced `1PN` corrections. It is not full N-body `1PN`.
- `solar-system-horizons` is the recommended preset for full Solar System central-body `1PN` experiments with stable body ordering and names.

Suggested validation workflow:

```bash
python nbody.py --preset mercury-relativistic --integrator rk4 --years 20 --dt 0.0001 --save-every 1 --save mercury_newtonian.npz --no-plot
python nbody.py --preset mercury-relativistic --gravity-model 1pn --pn-scope two-body --integrator rk4 --c 63239.7263 --years 20 --dt 0.0001 --save-every 1 --save mercury_1pn.npz --no-plot
python analyze_precession.py mercury_1pn.npz --compare mercury_newtonian.npz
```

Full Solar System central-body validation workflow:

```bash
python nbody.py --preset solar-system-horizons --horizons-epoch "2026-01-01 00:00" --integrator rk4 --years 5 --dt 0.0002 --save-every 5 --save solar_system_newtonian.npz --no-plot
python nbody.py --preset solar-system-horizons --horizons-epoch "2026-01-01 00:00" --gravity-model 1pn --pn-scope central-body --pn-primary-index 0 --integrator rk4 --c 63239.7263 --years 5 --dt 0.0002 --save-every 5 --save solar_system_1pn.npz --no-plot
python validate_planetary_precession.py solar_system_newtonian.npz solar_system_1pn.npz --primary-index 0
```

For the physically scaled Mercury case, the differential precession should be close to `43 arcsec/century`.

The orbit-comparison CLI reports:

- mean and standard deviation of semi-major axis
- mean and standard deviation of eccentricity
- periapsis-angle span
- precession-rate differences
- relative drift of the saved energy diagnostic

`validate_planetary_precession.py` reports, for each secondary:

- the baseline-subtracted precession rate (`1PN` minus Newtonian)
- the bundled canonical literature value when available
- the error relative to that literature value
- a formula-based Schwarzschild reference from bundled mean orbital elements
- for Pluto, the bundled comparison currently falls back to the formula-based reference because a canonical literature value is not bundled

`report_solar_system_validation.py` extends that workflow by also reporting:

- delta semi-major axis mean
- delta eccentricity mean
- per-body tolerance status for the bundled inner-planet validation targets
- a named validation profile header describing the intended run horizon

Compact-binary status:

- The built-in pulsar presets reproduce the standard `1PN`/EIH periastron-advance scale well enough for validation work.
- The Double Pulsar `1PN` EIH run remains the cleanest compact-binary validation target in the current codebase.
- The current `2PN` and `2.5PN` compact-binary workflows should be interpreted as controlled theory experiments rather than full timing-model reproductions.
- In particular, the present `2.5PN` path now includes the missing harmonic conservative `2PN` sector and still gets the correct dissipative sign in short compact-binary runs, but it is not yet a precision-grade timing-model implementation.


## Performance Considerations

- Complexity: O(N^2) per step (all-pairs force, vectorized with NumPy broadcasting)
- Practical ranges on typical laptops:
  - hundreds to a few thousand particles are reasonable
  - For much larger N, consider Barnes-Hut (O(N log N)), Particle-Mesh, or GPU acceleration
- Memory:
  - Positions and velocities are stored as (T, N, 3) if you save every frame; use --save-every to thin the output


## Output and Plotting

- The right subplot displays energy diagnostics:
  - dE/E0 = (E - E0) / |E0|
  - K and U are shown for reference
- In `1PN` mode, the right subplot switches to the relative drift of the two-body `1PN` total energy instead of the Newtonian `K/U/E` decomposition.
- In `2.5PN` mode, the right subplot shows the drift of the conservative bookkeeping energy; because radiation reaction is dissipative, that diagnostic is not a conserved quantity.
- The left subplot displays trajectories in either `2d` or `3d` mode.
- Use `--plot-mode 2d` for the XY projection or `--plot-mode 3d` for a full 3D trajectory view.
- To create animations, first save the simulation with `--save`, then run `render_video.py` on the resulting `.npz`.
- `.mp4` export requires an available `ffmpeg` writer in Matplotlib; `.gif` export requires the `Pillow` writer.


## Extending the Project

- GPU support (PyTorch): Replace NumPy ops with PyTorch tensors and support device=cpu|cuda.
- C++/Rust core: Keep the Python CLI and plotting and call a compiled kernel for the force computation.
- Better initial conditions: Plummer sphere, Keplerian N-planet systems, disks with Toomre Q, etc.
- Advanced integrators: e.g., Yoshida 4th-order (symplectic composition); adaptive or block timesteps (mind symplectic constraints).


## API Overview (Python)

```python
from nbody import (
    gravitational_acceleration, total_energy,
    two_body_sun_earth, three_body_figure8, star_earth_jupiter, random_nbody,
    Leapfrog, RK4, SimulationConfig, run_simulation
)

x0, v0, m, G = two_body_sun_earth()
cfg = SimulationConfig(dt=1e-3, steps=10000, save_every=10, eps=1e-3,
                       integrator='leapfrog', preset='two-body')
res = run_simulation(x0, v0, m, G, cfg)
```

res is a dict with arrays: times, traj, vel, K, U, energy and metadata.


## Troubleshooting

- Nothing appears on screen: add --plot, and ensure Matplotlib is installed.
- Energy drift is large: decrease --dt, try Leapfrog, and/or increase --eps modestly.
- `2.5PN` inspiral estimates are noisy: run longer integrations, reduce `--dt`, and compare first against leading-order Peters formulas rather than a full timing model.
- Run is slow at large N: expected with O(N^2); thin outputs via --save-every, or consider a treecode/GPU.


## License

MIT (or your preferred license). Add a LICENSE file if needed.

---

## Acknowledgments

- Classical N-body practice uses symplectic integrators (like Leapfrog) for Hamiltonian systems due to their superior long-term stability.
- The three-body figure-8 choreography ICs are the canonical set from Moore (1993) and Chenciner & Montgomery (2000).
