# Scaling study of the 2D reconnection code + a 3D line-tied kink-instability loop model

Files

| file | role |
|---|---|
| `tearing_theory.py` | independent linear tearing **eigenvalue solver** (same equations, same y-discretisation) + FKR/Coppi formulas |
| `scaling_study_2d.py`, `scaling_analysis.py` | parametric runs (linear benchmark, nonlinear S-scan, convergence) and the analysis/figure |
| `scaling_study_2d.png` | 9-panel summary of the 2D trends |
| `kink_loop_3d.py` | 3D line-tied **reduced-MHD** solver, zero-net-current twisted loop, anomalous resistivity, field-line tracing, coupling to the 1D loop |
| `kink_production.py`, `kink_figure.py` | production / refinement runs and the single 3D figure |
| `kink_loop_3d.png` | 12-panel 3D figure (field lines, helical current sheet, energy budget, threshold, robustness, 1D-loop response) |
| `scaling_results/*.json,npz` | all raw results (nothing below is hand-typed from memory of a run) |

---

## 0. Corrections to the previous deliverable (please read)

Running parametric variations exposed problems in `plasmoid_chain_coronal_loop.py`:

1. **The Harris sheet was not held in equilibrium**, so it diffused on `t ~ a²/η` (≈1.5 τ_A at S=500, ≈30 τ_A at
   S=10⁴), comparable to the run length. The "plasmoid chain" was dominated by islands imposed by the 24-mode seed
   (N_O = 32–36 at t=0), not selected by tearing. The earlier S-scan (6, 8, 8, 13 islands) and the claim
   "first chain at S≈500 / N∝S^{3/8}" are **not** evidence of plasmoid-chain onset.
2. **"P = I·Φ̇ confirmed (ratio 1–2.7)" was not a discriminating test**: `I_sheet` used an arbitrary threshold and `P` was
   whole-box Ohmic power. For a *held* sheet `P = I·E₀` holds trivially (E₀ = ηj₀ is the external field), so I dropped it
   rather than invent a better-looking test.
3. **Two citations were wrong**, now fixed everywhere: Servidio et al. 2010 is *Phys. Plasmas* **17**, 032315 (not JGR);
   Fuselier et al. 2022 is article **e2022JA030354**.

What *is* solid: the 2D solver reproduces the linear tearing eigenvalues (§1.1).

---

## 1. Parametric scalings of the 2D code — does it do what theory says?

Predictions were fixed before looking at output. Verdicts are from the numbers.

### 1.1 Verification: code vs an independent eigenvalue solver  ✅

The same linearised reduced-MHD equations about the same double-Harris equilibrium are solved as an eigenproblem
(`tearing_theory.py`). The nonlinear code is run with the equilibrium held fixed and a tiny multimode seed.

| η | N | fastest mode | code γ | eigen γ | error |
|---|---|---|---|---|---|
| 1e-3 | 128 | m=1 | 0.1935 | 0.1926 | +0.5 % |
| 4e-4 | 192 | m=1 | 0.1433 | 0.1434 | −0.0 % |
| 1.5e-4 | 256 | m=1 | 0.0978 | 0.0981 | −0.4 % |
| 6e-5 | 256 | m=1 | 0.0655 | 0.0656 | −0.1 % |
| 6e-5 | **192** | m=1 | 0.0623 | 0.0656 | **−5.1 %** (under-resolved) |

Modes within ~25 % of γ_max agree to 0.0–1.6 % (η ≥ 4e-4) but only to ~10–12 % for m=2 at η=6e-5 (contaminated by
the faster m=1 harmonic). Modes much slower than the fastest (m ≥ 4) are **not measurable** in a multimode run: they are
driven nonlinearly by faster modes (fitted "γ" up to 10²× too large) and are flagged, not reported. N=192→256 at η=6e-5
removes the 5 % error, i.e. the resolution requirement is visible and quantified.

### 1.2 Linear scaling of the tearing growth rate (eigenvalues, S_a = a v_A/η = 30 … 6000)  ✅ (approaches asymptote)

| quantity | theory (FKR/Coppi) | measured |
|---|---|---|
| γ_max · a/v_A ∝ S_a^p | p → −1/2 | global fit −0.36; **local slopes −0.21, −0.33, −0.39, −0.42, −0.43** |
| k_max · a ∝ S_a^q | q → −1/4 | global −0.17; S_a ≥ 300: **−0.22** |

The exponents converge monotonically toward the asymptotic values from above, which is the expected behaviour of
matched asymptotics (the −1/2 law holds for S_a ≫ 1; at S_a ≲ 10³ you are in the crossover). Claiming "−1/2" from
this range would be wrong; "approaching −1/2" is what the data show.

### 1.3 Onset time ∝ 1/γ_max  ✅

Nonlinear runs with the sheet held (so S_a is fixed), tiny random seeds (16 modes, 10⁻⁵ each). Onset = first time the
reconnected flux Ψ_rec reaches 0.05 (an absolute, N-independent definition; the earlier "20 % of the run maximum"
definition inherited the unconverged saturation level).

| S | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| t_onset [τ_A] | 7.8 | 8.6 | 9.8 | 13.0 | 17.0 |

Fitted exponent **0.284** vs **0.286** implied by the independent eigenvalue γ_max(S_a). Agreement is in the *local*
exponent; the asymptotic S^{1/2} is not yet reached in this range (§1.2).

### 1.4 Number of self-selected islands  ✅ (weak test)

Median island count after onset: 8, 8, 8, 8, 9 for S = 500 … 8000 vs the linear-theory prediction `2·k_max·L/2π = 8`
(k_max = 4 on the integer-mode grid, and it barely moves across this S range). Consistent, but **this is not a test of the
plasmoid-regime `N ∝ S^{3/8}`**: that law needs sheets with S_L ≳ 10⁴ resolved to δ ≪ ℓ, which this box cannot do.

### 1.5 Sweet–Parker thickness  ❌ not reproduced — and a trap worth knowing about

Plotting `δ/ℓ` against `S_ℓ = ℓ B_up/η` for 140 resolved X-point layers (after onset) gives a fitted slope of **−0.50**,
exactly the Sweet–Parker value. **That agreement is a coincidence.** Sweet–Parker requires
`δ = ℓ^{1/2} B_up^{−1/2} S^{−1/2}`, and a multi-variable regression on the same data gives

`δ ∝ ℓ^{−0.06} S^{−0.24} B_up^{+0.56}`   (SP: ℓ^{+0.5} S^{−0.5} B_up^{−0.5})

with the measured δ a median **4.3× thicker** than the Sweet–Parker value. The single-variable slope looks right
because ℓ grows with S (median 0.6 → 2.0), which drags `S_ℓ` and `δ/ℓ` along together. With only the three lowest S the
slope was −0.02; adding two more S values moved it to −0.50 without any change in the physics — a reminder not to trust a
log–log slope when the regressors are correlated. The sheet is held at thickness `a`, so δ ≳ a rather than thinning to
`δ_SP`. A separate probe of X-point collapse in a thick unheld sheet (S=2000, a=0.4, ε=0.03) thinned the layer only 1.6×
(j_X: 2.3 → 3.8; Sweet–Parker would need ~14). **The Sweet–Parker scaling is therefore not demonstrated.** A proper test
needs a larger seed, an unheld sheet and N ≳ 384.

### 1.6 Peak X-point rate  (partial)

`M = η j_X / B_up²` peaks at 0.035, 0.023, 0.014, 0.011, 0.0072 for S = 500 … 8000: slope **−0.55**. It sits above the
held-sheet baseline `1/S_a` (which would give slope −1), so the nonlinear stage does enhance the rate, but the number is
dominated by the sheet-sustaining field E₀ and a −0.55 slope is not evidence for Sweet–Parker (§1.5).

### 1.7 Energy closure ✅ and resolution convergence ⚠️

`E_B + E_K + Q_ohm + Q_visc − W_forcing` is conserved to ≤ 3.6×10⁻⁴ for all five runs (W_forcing = work done by the
external field that holds the sheet). The dissipation partition shifts with S: Ohmic fraction 0.93, 0.92, 0.89, 0.82, 0.70
(viscous 0.07 → 0.30) as S goes 500 → 8000 at Pm = 1.

Convergence at S=2000 (N = 128, 192, 256): time for Ψ_rec to reach 0.2 is 9.2, 13.2, 13.6 — **the early (linear → onset)
phase is converged for N ≥ 192 (3 %)**, but the end-of-run reconnected flux is 0.83, 0.66, 0.47 and the saturation speed
still changes: **the late nonlinear phase is not converged at N ≤ 256.** (Closure residual of these runs: 1.2×10⁻³,
6.6×10⁻⁴, 8×10⁻⁵ — improving with N.) Statements about onset and linear physics above are on firm ground; statements
about saturation level, island coalescence and the M_peak/partition numbers are resolution-limited. The scan used
N = 192, 256, 256, 384, 384 for S = 500 … 8000, which is marginal at the high end (δ_in/Δx ≈ 1).

---

## 2. 3D line-tied kink instability, zero net current

### 2.1 Physical model (and what is *not* Lare3D)

I did **not** write a Lagrangian-remap compressible code. Lare3D (Arber et al. 2001) solves the full resistive MHD
equations on a staggered grid with a Lagrangian step + remap, constrained transport for ∇·B=0 and shock viscosity.
Re-implementing that in NumPy at usable resolution is not feasible. Instead I used **3D reduced MHD** (Strauss 1976),
the asymptotic limit of a long thin flux tube in a strong axial field B₀ — the same family used for line-tied braided
loops in the Rappazzo/Parker papers cited by Cozzo et al. 2026:

```
dψ/dt = −[φ,ψ] + B₀ ∂φ/∂z + η(j) j          B = B₀ ẑ + ẑ×∇ψ,  v = ẑ×∇φ
dw/dt = −[φ,w] + [ψ,j] + B₀ ∂j/∂z + ν ∇²w    j = ∇²ψ,  w = ∇²φ
```

What I borrowed from Lare3D's kink studies (Hood et al. 2009; Bareford, Hood & Browning 2013):
**line-tying** (v = 0 at z=0,L), **aspect ratio L/R_b = 20**, **zero-net-current** loops, a velocity kick
`v_r = 0.01 e^{−4r⁴}…` to seed the kink, and **current-threshold anomalous resistivity**
(`η = η_c H(|j|−j_crit)`, η_c = 10⁻³ as in Lare3D's runs). What it lacks relative to Lare3D: compressibility, parallel
flows, pressure, gravity, shock viscosity, and non-slender (B_θ ≳ B_z) effects — Lare3D's Bessel-function loops have
B_θ/B_z ~ 0.7, outside the RMHD ordering, so thresholds are **not** quantitatively comparable.

Numerics: Fourier pseudo-spectral in (x,y) with 2/3 de-aliasing and integrating-factor RK4 for ν; **staggered**
finite differences in z (ψ, j at cell centres; φ, w at vertices including the footpoint planes). On this grid the two
`B₀∂/∂z` terms are exact discrete adjoints, so line-tied waves reflect correctly with no boundary closure, and
`E = ½∫(|∇ψ|²+|∇φ|²)dV` is conserved by the linear terms.

Equilibrium: `B_θ = λ r (1 − r⁶)²` (r<1, zero outside), `ψ(s)=(λ/2)[s − s⁴/2 + s⁷/7]`, `s=r²`; axis twist
`Φ₀ = λL`. Core current + return-current shell, enclosed current zero to 10⁻³ (verified numerically). Any
axisymmetric j(r) is an RMHD equilibrium because `[ψ, j(ψ)] = 0`.
A first profile `B_θ = λ r (1−r²)²` (twist concentrated at the axis) was **stable up to Φ₀ = 14π** — the
return-current shell stabilises it — so I broadened the twist plateau; this is a modelling choice, not a result about
real loops.

### 2.2 Verification of the 3D solver

| test | expected | result |
|---|---|---|
| line-tied Alfvén wave period | 2L/B₀ = 40 | **40.000** (error 0.00 %) |
| unseeded equilibrium static | E_k = 0 | max E_k = 1×10⁻⁶, ΔE_B/E_B = −7.5×10⁻⁶ over t=30 |
| initial energy | `πLλ²·0.0789 = 5.994` (Φ₀=7π) | 5.993 |
| ideal energy drift to nonlinear onset | small | ≲ 2 % of E_B at |j| ≈ 4–5 j_eq; ~10⁻⁴ in the linear phase |
| growth ∝ B₀/L (Φ₀ fixed) | t_nl ratio 2 for L=40 | **2.14** |

### 2.3 Threshold and sensitivity — read this before quoting any number

Time to nonlinearity (|j| = 2 j_eq), N=64, N_z=32, ν=2×10⁻⁴: 322, 286, 266 (Φ₀ = 4.0, 4.1, 4.15 π), 196 (4.25π),
124 (4.5π), 108 (4.75π), 100 (5π), 70 (6π), 52 (7π) τ_A. Extrapolating `1/t_nl → 0` gives
**Φ_c ≈ 3.6π** for this profile at this resolution. Every run at Φ₀ ≥ 4π is unstable; an early run that looked
"stable at 4π" was just not run long enough (growth is slow near threshold).

But the instability is **sensitive to numerical parameters**, and I do not claim convergence:

* ν = 10⁻³ (Φ₀=6π): **stable** by t=150; ν = 2×10⁻⁴: t_nl = 70; ν = 10⁻⁴: t_nl = 60.
* N = 96, N_z = 48 (Φ₀=6π): t_nl = 130 vs 70 at N=64, N_z=32 (×1.86).
* The growth-rate numbers near threshold are contaminated by the kick's oscillating transient (E_k ~ 10⁻⁴) and are good
  to ~30 %; only t_nl and stable/unstable are robust.

This behaviour (damping by ν, dependence on resolution) suggests the line-tied mode has fine radial/axial structure
(resonant layers), so the quantitative threshold needs a convergence study I only partly did (see the N/N_z/ν table in
the figure). Qualitatively consistent with the literature: the zero-net-current return shell raises the threshold above
the 2.5π of an uniform-twist loop with net current (Hood & Priest 1979/1981); the e-folding time of tens of τ_A and
onset ≈ 50–100 τ_A match the Lare3D runs of Bareford et al. 2013 (onset ≈ 40–50 τ_A, ~70 % of the release within the
next 50 τ_A).

**The kink onset is NOT converged in the perpendicular resolution — and at N=128 it disappears** (Φ₀ = 7π; time to
|j| = 2 j_eq, ν = 2×10⁻⁴ unless noted):

| N | N_z | ν | t_nl | note |
|---|---|---|---|---|
| 64 | 32 | 2e-4 | 52 | baseline of the threshold scan |
| 64 | 48 | 2e-4 | 50 | axial resolution **converged** (−4 %) |
| 96 | 32 | 2e-4 | 150 (84 to 1.5 j_eq) | onset delayed ×2.9 |
| **128** | 32 | 2e-4 | **no growth by t = 170** (|j|_max = 1.34 j_eq, E_k = 1.3e-4) | |
| 64 | 32 | 1e-3 | 118 | viscosity delays the onset ×2.3 |
| 64 | 32 | 1e-4 (6π) | 60 vs 70 | |

Follow-up runs at N = 128 (same equilibrium, `kink_conv3d_b.json`) settle the interpretation:

| N | Φ₀ | ν | t_nl | outcome |
|---|---|---|---|---|
| 128 | 7π | 2e-4 | — | no growth by t = 170 |
| 128 | 7π | 1e-4 | 116 | **unstable** |
| 128 | 7π | 5e-5 | 88 | **unstable** |
| 128 | 10π | 2e-4 | 98 | **unstable** |

**So the kink is a real mode, not a numerical artefact — but its fine radial/axial structure is damped by a *resolved*
viscosity.** At N = 128 the 7π loop is stable at ν = 2×10⁻⁴ yet unstable at ν = 10⁻⁴ and 5×10⁻⁵ (onset 116 → 88 τ_A,
approaching the ideal value from above as ν falls), and a stronger twist (10π) is unstable even at ν = 2×10⁻⁴. The coarse
grids (N ≤ 96) cannot represent the mode's thin structure, so they under-resolve its viscous damping and over-predict
instability at a given ν. Consequences:

* The **N = 64 threshold Φ_c ≈ 3.6π is not valid** (it is an under-resolved, viscosity-blind number). At ν = 2×10⁻⁴ the
  converged threshold lies above 7π; its ideal-limit value (ν → 0) was not determined.
* The **N = 64 onset times (52 τ_A at 7π) are too short**; the better-resolved, low-viscosity value is ≈ 88 τ_A — close to
  the Lare3D onset of ≈ 50–100 τ_A, but I would not call that agreement quantitative.
* The **N = 80, ν = 2×10⁻⁴, 7π event of §3 is unstable only because the grid is coarse** (the same loop is stable at
  N = 128, ν = 2×10⁻⁴). Its end-state diagnostics are still meaningful as a property of the nonlinear relaxation, but it
  is not a converged loop. The resolved counterpart is the 10π, N = 128 run (§3.2).

What survives: (a) the solver verification (§2.2); (b) the qualitative stabilisation by the return-current shell (the
p = 1 profile, whose twist is concentrated at the axis, stays stable up to 14π at N=64); (c) the 1/L scaling of the
kink time (×2.14 for L=40 at fixed dz, expected ×2); (d) the *end-state* physics once an instability has run (§3).

---

## 3. Nonlinear kink run (Φ₀ = 7π, N=80, N_z=40, ν=2×10⁻⁴, anomalous η_c=10⁻³ above j_crit = 1.8 j_eq,max)

Everything below is read from `scaling_results/kink_summary.json` (produced by `kink_figure.py`).

| quantity | value | comparison |
|---|---|---|
| ramp growth rate (E_k 10⁻³ → 4×10⁻², t = 40–65) | γ = 0.069 τ_A⁻¹ | e-fold ~14 τ_A; N=64 scan gave 0.103 at 7π (resolution-dependent) |
| onset of nonlinearity (|j| = 2 j_eq) | t ≈ 65 τ_A | Lare3D Loop B: ≈ 40–50 τ_A (Bareford+13) |
| E_k peak | 0.81 at t = 90 | |
| magnetic energy released | **4.79 of 5.99 = 79.9 %** of the tube's E_B | |
| timing | 89 % of it released in the 45 τ_A after onset | Lare3D: ~70 % in the next 50 τ_A |
| heating partition (Ohmic : viscous) | **0.50 : 0.50** (2.40 : 2.39) | Lare3D (η_b=0): mostly shock-viscous |
| energy accounted for as heat + E_k | **101 %** of ΔE_B (budget residual 1.1 %) | Lare3D: 68 % (η_b=0), 96 % (η_b=10⁻⁴) |
| max current | 4.2 × j_eq | |
| **magnetic helicity** K = 2B₀∫(ψ_∞−ψ)dV | K_final/K₀ = **1.011** (max deviation 1.2 %) | Lare3D: change ≤ 2 % |
| **Taylor partial relaxation** | R_l implied by (K_f, W_f) = **1.344**; measured uniform-α-equivalent radius = **1.332**; predicted energy release at the measured radius **79.6 %** vs simulated **79.9 %** | Bareford+13: partial relaxation, loop expands 1.5–1.8 R_b |

The relaxation check is not circular: κ = 4K/(πLB₀) comes from the *initial* helicity and the radius from the *final*
field's r_rms (`r_rms² = (2/3)R_l²` for a uniform-α state); the predicted release `ΔW/W₀ = 1 − πLκ²/(16 R_l⁴ W₀)` is then
compared with the simulated one. Agreement to 0.3 percentage points says the end state is close to a helicity-conserving
linear force-free state, as in the Lare3D papers. Reading it correctly: it supports the **end-state** quantities (release
fraction, helicity, radius), which relaxation theory predicts independently of dissipation details; it does **not**
rescue the onset time or threshold, which are resolution-dependent (§2.3).

Honest limits of this run:
* The nonlinear stage is **under-resolved**: at t = 90 the current shows grid-scale speckle (panels c, d are therefore
  shown at t = 72, the onset of nonlinearity, where the helical structure is coherent). The 50:50 Ohmic/viscous split
  depends on η_c, ν and N and should not be taken as physical.
* Energy closure is good *because the dissipation is explicit* (ν w², η j²); unresolved structure shows up as the
  1.1 % residual, not as a hidden loss as in Lare3D's η_b = 0 runs.

### 3.1 Field-line coupling to the 1D loop (Reid et al. 2021 procedure)

For 48 footpoints the field line is traced through every saved snapshot, the heating `η(j)j² + νw²` sampled along it and
converted to SI with Reid et al.'s reference values (B = 10 G, n = 10¹⁵ m⁻³ → v_A = 690 km/s, strand radius 4.5 Mm,
L = 90 Mm; 1 code energy unit = 7.25×10¹⁹ J). It drives the field-aligned loop of the previous study (gravity,
Spitzer conduction, KPC08 radiation, chromospheric reservoir) after a 1500 s relaxation with uniform heating that is then
switched off, exactly as Reid et al. do.

| field line | apex T before → peak | peak parallel flow | peak apex density |
|---|---|---|---|
| axis | 1.8 → **3.9 MK** | 194 km/s | 4.2×10¹⁴ m⁻³ |
| r = 0.6 | 1.8 → **2.4 MK** | 82 km/s | 1.9×10¹⁴ m⁻³ |
| most heated | 1.8 → **4.8 MK** | 224 km/s | 3.6×10¹⁴ m⁻³ |

Released energy 3.5×10²⁰ J (3.5×10²⁷ erg): a whole strand kinking is ~10³ × a nanoflare, i.e. microflare scale;
nanoflare-size events need sub-strand kinks. The response is a single event (no sustained heating), so the loop cools
afterwards. No feedback of the density on the MHD is included (Reid et al. discuss this limitation too).

---

## 4. What this means for the 3D results (summary of reliability)

| claim | status |
|---|---|
| solver correctness (line-tied Alfvén period, static equilibrium, energy, 1/L scaling) | ✅ verified |
| zero-net-current return shell stabilises: p = 1 profile stable to 14π | ✅ (N = 64; direction robust, number not) |
| kink onset time, growth rate, threshold at N ≤ 96 | ❌ not converged (resolution- and viscosity-dependent) |
| kink exists and is damped by resolved viscosity | ✅ N = 128 runs (§2.3) |
| ν → 0 (ideal) threshold | ❓ not determined |
| end-state: release fraction, helicity conservation, relaxed radius, partition | ✅ qualitatively robust; the Ohmic:viscous split is not |
| 1D-loop response | ✅ procedure; amplitudes depend on assumed B, n, a and are single-event |

## 5. Running it

```bash
python tearing_theory.py                       # eigenvalue solver demo
python scaling_study_2d.py all                 # theory + linear benchmark + nonlinear scan + convergence (hours)
python scaling_analysis.py                     # -> scaling_study_2d.png, scaling_results/scaling_summary.json
python kink_loop_3d.py --test                  # Alfven-wave and static-equilibrium verification
python kink_loop_3d.py --scan-p3               # twist scan (p=3 profile)
python kink_production.py prod                 # nonlinear kink with snapshots (~30-100 min)
python kink_production.py refine | refine2 | conv3d | conv3d_b   # threshold / convergence jobs
python kink_figure.py                          # -> kink_loop_3d.png, scaling_results/kink_summary.json
```

Requirements: `numpy`, `scipy`, `matplotlib`; `imageio-ffmpeg` is not needed here. Set `MHD_FFT_WORKERS` to limit FFT
threads when running several jobs at once.

---

## References

Verified against the source during this work (PDF text, DOI page, or search):
* Hood, Browning & Van der Linden 2009, *A&A* 506, 913 — coronal heating by reconnection in zero-net-current loops (Lare3D).
* Bareford, Hood & Browning 2013, *A&A* 550, A40 — partial relaxation of twisted loops (read in full; L=20 R_b, line-tied,
  η_c=10⁻³ above J_crit, 68 %/96 % energy accounting, Table 1 loops).
* Browning, Gerrard, Hood, Kevis & Van der Linden 2008, *A&A* 485, 837; Gerrard & Hood 2003, *Solar Phys.* 214, 151;
  Bareford, Browning & Van der Linden 2010 *A&A* 521, A70 and 2011 *Solar Phys.* 273, 93; Baty 2000 *A&A* 360, 345;
  Botha, Arber & Hood 2011 *A&A* 525, A96; Evans & Hawley 1988 *ApJ* 332, 659; Taylor 1974 *PRL* 33, 1139
  (all via the Bareford et al. reference list).
* Arber, Longbottom, Gerrard & Milne 2001, *J. Comput. Phys.* 171, 151 (Lare3D).
* Hood & Priest 1979, *Solar Phys.* 64, 303 (doi 10.1007/BF00151441).
* Servidio, Matthaeus, Shay, Dmitruk, Cassak & Wan 2010, *Phys. Plasmas* 17, 032315.
* Reid, Cargill, Johnston & Hood 2021, *MNRAS* 505, 4141; Cozzo et al. 2026, *ApJ* 998, 76; Longcope & Tarr 2015,
  *Phil. Trans. R. Soc. A* 373, 20140263; Pankin et al. 2025, *Comput. Phys. Commun.* 312, 109611; Fuselier et al. 2022,
  *JGR Space Phys.* 127, e2022JA030354 (the five PDFs supplied).

