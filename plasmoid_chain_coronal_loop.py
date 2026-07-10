
r"""
Plasmoid-chain reconnection + field-line-coupled 1D coronal loop
================================================================

An UPDATED model that pushes the 2D resistive-MHD reconnection code
(`mhd_reconnection_coronal_heating.py`) toward higher Lundquist number S (higher
resolution N, lower resistivity eta) until a thin current sheet fragments into a
chain of magnetic islands (a *plasmoid chain*), and then adds four physically
grounded diagnostics/couplings that the original passive-scalar model lacked:

  (1) X-LINE / O-POINT COUNTING  -- automatic detection of reconnection sites
      (X-points) and islands/plasmoids (O-points) as critical points of the
      flux function, counted vs. S and vs. time.
  (2) RECONNECTION RATE  Phi_dot  -- the rate of flux transfer measured as the
      out-of-plane electric field at the dominant X-point, E_rec = eta*j_X =
      -d(psi_X)/dt, plus the dimensionless rate M = E_rec/(v_A B_up).
  (3) EXPLICIT  P = I * Phi_dot  TEST  -- Longcope & Tarr (2015): the ratio of
      dissipated power to reconnection rate is a current; we verify
      I = P_ohm / Phi_dot  against the current actually threading the sheet.
  (4) FIELD-LINE COUPLING TO A 1D LOOP  -- the local Ohmic+viscous heating is
      sampled along a field line and fed as Q(s,t) into a Reid-style 1D
      field-aligned hydrodynamic loop with gravity, Spitzer conduction, optically
      thin radiation and a chromospheric mass reservoir -- i.e. a real
      thermodynamic loop response (evaporation/draining), NOT a passive scalar.

Physical grounding & references
-------------------------------
Resistive MHD / tearing / plasmoid instability
  [T1] Furth, Killeen & Rosenbluth 1963, Phys. Fluids 6, 459      (tearing mode)
  [T2] Biskamp 1986, Phys. Fluids 29, 1520                        (sheet -> plasmoids)
  [T3] Loureiro, Schekochihin & Cowley 2007, Phys. Plasmas 14, 100703
                                                (plasmoid instability, S_c ~ 1e4)
  [T4] Samtaney, Loureiro, Uzdensky, Schekochihin & Cowley 2009, PRL 103, 105004
                                                (N_plasmoids ~ S^{3/8})
  [T5] Bhattacharjee, Huang, Yang & Rogers 2009, Phys. Plasmas 16, 112102
                                                (fast plasmoid-dominated reconnection)
  [T6] Uzdensky, Loureiro & Schekochihin 2010, PRL 105, 235002    (stochastic chain)
  [T7] Comisso, Lingam, Huang & Bhattacharjee 2016, Phys. Plasmas 23, 100702 (onset)
Reconnection diagnostics
  [D1] Servidio et al. 2010, J. Geophys. Res. 115, A05111  (X/O critical points in MHD)
  [D2] Fuselier et al. 2022, J. Geophys. Res. Space Phys. 127, e2022JA030281
                                                (MULTIPLE reconnection X-lines)
  [D3] Longcope & Tarr 2015, Phil. Trans. R. Soc. A 373, 20140263   (P = I*Phi_dot)
  [D4] Pankin et al. 2025, Comput. Phys. Commun. 312, 109611 (TRANSP: poloidal-flux
                                     diffusion & Ohmic power P = I*V_loop analog)
Coronal-loop hydrodynamics / heating
  [L1] Reid, Cargill, Johnston & Hood 2021, MNRAS 505, 4141  (3D MHD heating ->
                                     1D field-aligned loop: THE method reproduced here)
  [L2] Cozzo et al. 2026, ApJ 998, 76   (braiding loop; anomalous eta above J_crit;
                                     Ohmic + conduction + radiation + chromosphere)
  [L3] Klimchuk, Patsourakos & Cargill 2008, ApJ 682, 1351  (radiative loss function)
  [L4] Johnston & Bradshaw 2019, ApJ 873, L22; Johnston et al. 2020, A&A 635, A168 (TRAC)
  [L5] Spitzer 1962, "Physics of Fully Ionized Gases"        (kappa0 T^{5/2})
  [L6] Hood, Cargill, Browning & Tam 2016, ApJ 817, 5        (MHD avalanche)
  [L7] Arber, Longbottom, Gerrard & Milne 2001, JCP 171, 151 (Lare)
  [L8] Parker 1988, ApJ 330, 474                             (nanoflares)
Reviews
  [R1] Priest & Forbes 2000, "Magnetic Reconnection"; Biskamp 2000, "Magnetic
       Reconnection in Plasmas"; Zweibel & Yamada 2009, ARA&A 47, 291.

Run:  python plasmoid_chain_coronal_loop.py            (full production run)
      python plasmoid_chain_coronal_loop.py --scan     (only the S-scan)
      python plasmoid_chain_coronal_loop.py --quick     (fast, low-res preview)
"""

from __future__ import annotations

import sys
import time
import numpy as np

# Re-use the validated spectral resistive-MHD solver from the first model.
import mhd_reconnection_coronal_heating as base
from mhd_reconnection_coronal_heating import ReconnectionMHD, coronal_parameters

import matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.pyplot as plt


# ============================================================================
# 1.  High-S plasmoid-chain MHD: thin current sheet + broadband tearing seed
# ============================================================================
class PlasmoidConfig(base.SimulationConfig):
    """High-Lundquist-number configuration.

    A THIN current sheet (small `a`) is linearly unstable to tearing over a wide
    band of wavenumbers k with k*a < 1 [Furth, Killeen & Rosenbluth 1963 [T1]].
    Seeding many of those modes lets the fastest-growing finite-k mode win, so
    the sheet fragments into a *chain* of islands rather than a single island.
    Lowering eta (raising S) shortens the tearing wavelength and increases the
    number of plasmoids -- the hallmark of the plasmoid regime,
    N_plasmoids ~ S^{3/8} [Samtaney et al. 2009 [T4]; Loureiro et al. 2007 [T3]].
    """
    N = 512              # resolution (must resolve the thin sheet, a/dx >~ 4)
    L = 2.0 * np.pi
    a = 0.06             # THIN sheet half-width  (=> many unstable modes)
    eta = 1.0e-4         # resistivity -> S = 1/eta = 1e4 (plasmoid threshold)
    nu = 1.0e-4          # viscosity (Pm = 1)
    kappa = 3.0e-4       # MHD-side heat-field diffusivity (Ohmic proxy)
    seed_modes = 24      # number of Fourier modes in the broadband tearing seed
    seed_amp = 6.0e-3    # total rms amplitude of the seed
    vpert = 0.0          # start from rest (tearing, not driven)
    t_end = 24.0         # Alfven times (chain forms during nonlinear tearing)
    cfl = 0.30
    dt_max = 6.0e-3
    n_frames = 120
    seed_rng = 12345     # deterministic broadband phases


class PlasmoidMHD(ReconnectionMHD):
    """Doubly-periodic reduced-MHD solver seeded to make a plasmoid chain."""

    def _initial_condition(self):
        cfg = self.cfg
        L, a = cfg.L, cfg.a
        x, y = self.x, self.y

        # Double Harris sheet (periodic): Bx reverses across y=L/4 and y=3L/4.
        # A thin sheet (small a) has a large tearing instability parameter Delta'
        # and a short fastest-growing wavelength -> a chain of islands [T1,T3].
        Bx = (np.tanh((y - 0.25 * L) / a) - np.tanh((y - 0.75 * L) / a))
        Bx -= Bx.mean()
        Bx_hat = self._fft(Bx)
        psi_hat = np.zeros_like(Bx_hat)
        nz = self.ky != 0
        psi_hat[nz] = -Bx_hat[nz] / (1j * self.ky[nz])   # psi from Bx=-dpsi/dy

        # BROADBAND tearing seed: superpose many wavelengths with pseudo-random
        # phases so the linearly fastest-growing finite-k mode selects itself and
        # sets the number of plasmoids (rather than us imposing it).
        rng = np.random.default_rng(cfg.seed_rng)
        kx0 = 2.0 * np.pi / L
        gy = (np.exp(-((y - 0.25 * L) / (2 * a))**2)
              + np.exp(-((y - 0.75 * L) / (2 * a))**2))
        dpsi = np.zeros_like(y)
        M = cfg.seed_modes
        for m in range(1, M + 1):
            phase = rng.uniform(0, 2 * np.pi)
            dpsi += np.cos(m * kx0 * x + phase)
        dpsi *= cfg.seed_amp / np.sqrt(M) * gy
        psi_hat += self._fft(dpsi) * self.dealias

        w_hat = np.zeros_like(psi_hat)                 # start from rest
        T_hat = self._fft(np.ones_like(y))             # cool background
        return psi_hat, w_hat, T_hat


def make_mhd(cfg):
    return PlasmoidMHD(cfg)


# ============================================================================
# 2.  X-line / O-point counting diagnostic
# ============================================================================
# In 2D the in-plane magnetic field is B = z_hat x grad(psi), so its nulls are
# the CRITICAL POINTS of the flux function psi (grad psi = 0).  Classifying each
# null by the Hessian of psi separates the two reconnection topologies
# [Servidio et al. 2010 [D1]]:
#     det(Hessian) < 0  ->  SADDLE  ->  X-point  (a reconnection site / X-line)
#     det(Hessian) > 0  ->  EXTREMUM ->  O-point  (an island / plasmoid centre)
# Counting X-points therefore counts reconnection X-lines (cf. the MULTIPLE
# magnetopause X-lines of Fuselier et al. 2022 [D2]); counting O-points counts
# plasmoids.  A single Harris neutral line (one broad current sheet) carries no
# isolated nulls; once it tears into a chain, N_X and N_O jump together.
def critical_points(sim, psi, sheet_halfwidth=6.0, merge_cells=4.0,
                    smooth_cells=1.5, det_frac=0.015):
    """Locate and classify the nulls of the in-plane field (critical pts of psi).

    A genuine null needs BOTH partial derivatives to vanish, so we find the
    intersections of the two nullclines d(psi)/dx = 0 and d(psi)/dy = 0: a grid
    cell contains a critical point iff grad_x changes sign AND grad_y changes
    sign across the local 2x2 block.  This yields ISOLATED points even along a
    thin neutral sheet (where |grad psi| is small everywhere but the two
    nullclines cross only at discrete O/X points).  Each null is classified by
    the Hessian determinant of psi [Servidio et al. 2010 [D1]]:
        det H < 0 -> X-point (saddle / reconnection X-line)
        det H > 0 -> O-point (island / plasmoid centre)

    Two filters suppress spurious detections from grid-scale noise in the nearly
    field-free sheet: (i) a light Gaussian spectral low-pass of psi (`smooth_cells`
    in grid cells); (ii) a NON-DEGENERACY cut keeping only nulls whose |det H|
    exceeds `det_frac` of the strongest null (a real X/O point has finite field
    curvature; a flat-region false null does not).  Note tr(H) = laplacian(psi)
    = j, so the Hessian is set by the local current -- the physics ties the two.
    """
    ph = sim._fft(psi)
    if smooth_cells > 0:                       # Gaussian low-pass to kill ripple
        ph = ph * np.exp(-0.5 * sim.k2 * (smooth_cells * sim.dx) ** 2)
    gx = sim._ifft(sim._ddx(ph))
    gy = sim._ifft(sim._ddy(ph))
    pxx = sim._ifft(-sim.kx**2 * ph)
    pyy = sim._ifft(-sim.ky**2 * ph)
    pxy = sim._ifft(-sim.kx * sim.ky * ph)
    det = pxx * pyy - pxy**2
    jabs = np.abs(sim._ifft(sim.j_hat(ph)))

    def block_has_signchange(s):
        # True on cell (i,j) if the 2x2 block {(i,j),(i+1,j),(i,j+1),(i+1,j+1)}
        # contains both signs of s (periodic wrap)
        r = np.roll(s, -1, axis=0)
        u = np.roll(s, -1, axis=1)
        d = np.roll(np.roll(s, -1, axis=0), -1, axis=1)
        mn = np.minimum(np.minimum(s, r), np.minimum(u, d))
        mx = np.maximum(np.maximum(s, r), np.maximum(u, d))
        return (mn < 0) & (mx > 0)

    cell = block_has_signchange(np.sign(gx)) & block_has_signchange(np.sign(gy))

    # restrict to bands around the two current sheets (periodic in y)
    L, a = sim.cfg.L, sim.cfg.a
    band = ((np.abs(sim.y - 0.25 * L) < sheet_halfwidth * a) |
            (np.abs(sim.y - 0.75 * L) < sheet_halfwidth * a))
    cell &= band

    # non-degeneracy cut: drop flat-region false nulls (|det H| ~ 0)
    if det_frac > 0 and cell.any():
        det_cut = det_frac * np.abs(det[cell]).max()
        cell &= (np.abs(det) > det_cut)

    ii, jj = np.where(cell)
    if ii.size == 0:
        return dict(x_X=np.array([]), y_X=np.array([]), psi_X=np.array([]),
                    j_X=np.array([]), x_O=np.array([]), y_O=np.array([]),
                    psi_O=np.array([]), n_X=0, n_O=0)
    xs, ys, ps, ds = sim.x[ii, jj], sim.y[ii, jj], psi[ii, jj], det[ii, jj]
    gmag2 = (gx**2 + gy**2)[ii, jj]
    js = jabs[ii, jj]

    # greedy merge of neighbouring flagged cells (one null can trip a few cells)
    order = np.argsort(gmag2)
    keep, taken = [], np.zeros(order.size, bool)
    merge_r2 = (merge_cells * sim.dx) ** 2
    # map from position in `order`... use coordinate distance directly
    xa, ya = xs, ys
    done = np.zeros(xs.size, bool)
    for idx in order:
        if done[idx]:
            continue
        keep.append(idx)
        d2 = (xa - xa[idx])**2 + (ya - ya[idx])**2
        # handle periodicity in x
        dxp = np.minimum(np.abs(xa - xa[idx]), L - np.abs(xa - xa[idx]))
        d2 = dxp**2 + (ya - ya[idx])**2
        done |= (d2 < merge_r2)
    keep = np.array(keep, int)

    is_X = ds[keep] < 0
    is_O = ds[keep] > 0
    return dict(
        x_X=xs[keep][is_X], y_X=ys[keep][is_X], psi_X=ps[keep][is_X],
        j_X=js[keep][is_X],
        x_O=xs[keep][is_O], y_O=ys[keep][is_O], psi_O=ps[keep][is_O],
        n_X=int(is_X.sum()), n_O=int(is_O.sum()),
    )


# ============================================================================
# 3.  Reconnection rate Phi_dot  and the explicit  P = I * Phi_dot  test
# ============================================================================
def _sample(field, sim, xq, yq):
    """Bilinear sample of a grid field at physical coords (xq,yq), periodic."""
    from scipy.ndimage import map_coordinates
    ix = np.atleast_1d(xq) / sim.dx
    iy = np.atleast_1d(yq) / sim.dx
    return map_coordinates(field, [ix, iy], order=1, mode="wrap")


def reconnection_diagnostics(sim, f, cp, j_frac=0.25):
    r"""Measure the reconnection rate and run the Longcope-Tarr power test.

    RECONNECTION RATE.  At an X-point the in-plane field and the flow both
    vanish (it is a stagnation null), so the induction equation
    d(psi)/dt = -[phi,psi] + eta*lap(psi) collapses to d(psi)/dt = eta*j there.
    Hence the out-of-plane electric field E_z = eta*j_X IS the rate at which flux
    is transferred across the X-line -- the reconnection rate:
            Phi_dot = E_rec = eta * j_X            [Priest & Forbes 2000 [R1]].
    The dimensionless rate is M = E_rec/(B_up v_A,up); with v_A,up = B_up (rho=1)
    this is M = eta*j_X / B_up^2.  Sweet-Parker gives M ~ S^{-1/2}; plasmoid-
    mediated reconnection saturates near M ~ 0.01, independent of S
    [Bhattacharjee et al. 2009 [T5]; Uzdensky et al. 2010 [T6]].

    P = I * Phi_dot TEST  [Longcope & Tarr 2015 [D3]].  They observe that the
    ratio of dissipated power to reconnection rate is a CURRENT,
            I = P / Phi_dot ,
    and that it matches the current threading the reconnecting boundary.  We
    verify this directly: P = eta * integral(j^2 dA) (total Ohmic heating),
    Phi_dot = eta*j_X (dominant X-line EMF / "voltage"), so I_derived = P/Phi_dot
    should equal the measured sheet current I_sheet = integral(|j| dA) over the
    reconnecting layer.  This is the coronal analogue of the tokamak transformer
    relation P_ohmic = I_p * V_loop with V_loop = d(psi_pol)/dt, as solved by the
    poloidal-flux-diffusion / power-balance modules of TRANSP [Pankin et al.
    2025 [D4]].  Summing over the many X-lines of a plasmoid chain is exactly the
    multiple-X-line accounting seen at the magnetopause [Fuselier et al. 2022 [D2]].
    """
    eta = sim.cfg.eta
    j = f["j"]
    dA = sim.dx**2
    S = 1.0 / eta

    # total Ohmic dissipation  P = eta * integral j^2 dA
    P_ohm = eta * np.sum(j**2) * dA
    # current threading the reconnecting layer  I = integral |j| dA
    jmax = np.abs(j).max()
    sheet = np.abs(j) > j_frac * jmax
    I_sheet = np.sum(np.abs(j)[sheet]) * dA

    if cp["n_X"] == 0:
        return dict(S=S, P_ohm=P_ohm, I_sheet=I_sheet, E_rec=0.0, Phi_dot=0.0,
                    M=0.0, M_SP=1.0 / np.sqrt(S), Bup=1.0, I_derived=0.0,
                    ratio=0.0, xX=np.nan, yX=np.nan, jX=0.0)

    # dominant reconnection site = X-point of strongest current
    k = int(np.argmax(cp["j_X"]))
    xX, yX, jX = cp["x_X"][k], cp["y_X"][k], cp["j_X"][k]
    E_rec = eta * jX                                   # = Phi_dot (dominant X-line)

    # upstream reconnecting field: |B| a few sheet-thicknesses into the inflow
    Bmag = np.sqrt(f["Bx"]**2 + f["By"]**2)
    a = sim.cfg.a
    Bup = float(max(_sample(Bmag, sim, xX, yX + 3 * a)[0],
                    _sample(Bmag, sim, xX, yX - 3 * a)[0]))
    M = E_rec / (Bup * Bup + 1e-30)                    # normalized rate (v_A=B_up)

    # explicit P = I * Phi_dot  =>  I_derived = P / Phi_dot,  compare to I_sheet
    I_derived = P_ohm / (E_rec + 1e-30)
    ratio = I_derived / (I_sheet + 1e-30)

    return dict(S=S, P_ohm=P_ohm, I_sheet=I_sheet, E_rec=E_rec, Phi_dot=E_rec,
                M=M, M_SP=1.0 / np.sqrt(S), Bup=Bup, I_derived=I_derived,
                ratio=ratio, xX=xX, yX=yX, jX=jX)


# ============================================================================
# 4.  Field-line coupling to a 1D field-aligned coronal-loop hydro model
# ============================================================================
# The 2D MHD "temperature" is a passively advected scalar -- it has no thermal
# conduction, radiation, gravity or chromosphere, so it cannot tell us the real
# coronal response to reconnection heating.  Following Reid, Cargill, Johnston &
# Hood 2021 [L1], we instead SAMPLE the local Ohmic + viscous heating along a
# magnetic field line and feed it as Q(s,t) into a proper 1D field-aligned
# hydrodynamic loop that DOES include those processes [Cozzo et al. 2026 [L2]]:
#
#   d(rho)/dt      + d(rho v)/ds       = 0                         (mass)
#   d(rho v)/dt    + d(rho v^2 + p)/ds = rho g_par(s)             (momentum)
#   d(E)/dt        + d((E+p)v)/ds      = rho g_par v + Q           (energy)
#                                        + d/ds(kappa0 T^5/2 dT/ds)  (Spitzer cond.)
#                                        - n_e n_H Lambda(T)         (radiation)
#
# with a gravitationally stratified CHROMOSPHERE at both footpoints acting as a
# mass reservoir (evaporation/draining).  kappa0 T^{5/2} is Spitzer-Harm parallel
# conduction [Spitzer 1962 [L5]]; Lambda(T) is the optically-thin radiative loss
# function of Klimchuk, Patsourakos & Cargill 2008 [L3].

_KB = 1.380649e-23           # J/K
_MP = 1.67262192369e-27      # kg
_G_SUN = 274.0               # m/s^2 (solar surface gravity)
_KAPPA0 = 1.0e-11            # Spitzer parallel conduction, W m^-1 K^-7/2 [L5]
_GAMMA = 5.0 / 3.0


def radiative_loss(T):
    """Optically-thin radiative loss function Lambda(T) in SI (W m^3).

    Piecewise power law Lambda = chi * T^alpha with the coefficients tabulated by
    Klimchuk, Patsourakos & Cargill 2008 [L3] (as used in EBTEL); given here in
    cgs (erg cm^3 s^-1) and converted to SI by x1e-13.  The radiative loss per
    unit volume is then n_e n_H Lambda(T)  [W m^-3], with n in m^-3.  Below the
    transition region the losses are tapered to mimic the optically thick
    chromosphere (Reid et al. 2021 [L1]).
    """
    T = np.asarray(T, float)
    logT = np.log10(np.maximum(T, 1.0))
    # (log T upper bound, chi_cgs, alpha)
    segs = [(4.97, 1.09e-31, 2.00),
            (5.67, 8.87e-17, -1.00),
            (6.18, 1.90e-22, 0.00),
            (6.55, 3.53e-13, -1.50),
            (6.90, 3.46e-25, 1.0 / 3.0),
            (7.63, 5.49e-16, -1.00),
            (100.0, 1.96e-27, 0.50)]
    Lam = np.zeros_like(T)
    lo = -np.inf
    for hi, chi, alpha in segs:
        m = (logT > lo) & (logT <= hi)
        Lam[m] = chi * T[m]**alpha
        lo = hi
    Lam *= 1.0e-13                                  # erg cm^3 s^-1 -> W m^3
    # chromospheric taper below ~2e4 K (optically thick): reduce losses smoothly
    Ttap = 2.0e4
    taper = np.clip(T / Ttap, 0.0, 1.0)**2
    Lam = np.where(T < Ttap, Lam * taper, Lam)
    return Lam


class Loop1D:
    """1D field-aligned coronal-loop hydrodynamics (Reid-style [L1])."""

    def __init__(self, L_loop=50.0e6, n_cells=400, chromo_frac=0.12,
                 T_cor=0.8e6, n_cor=1.0e15, T_chr=1.0e4, n_chr=5.0e17):
        self.L = L_loop
        self.N = n_cells
        self.ds = L_loop / n_cells
        self.s = (np.arange(n_cells) + 0.5) * self.ds
        self.gamma = _GAMMA

        # semicircular-loop field-aligned gravity: height z(s) ~ sin, so the
        # component along the loop is g_par = -g_sun cos(pi s/L)  (toward the
        # nearer footpoint) -- zero at the apex, +-g_sun at the footpoints.
        self.g_par = -_G_SUN * np.cos(np.pi * self.s / L_loop)

        # chromosphere occupies the outer `chromo_frac` at each footpoint
        s_chr = chromo_frac * L_loop
        self.chromo = (self.s < s_chr) | (self.s > L_loop - s_chr)
        self.corona = ~self.chromo

        # initial atmosphere: cool dense chromosphere + warm corona, roughly
        # hydrostatic; the code then relaxes it under conduction/radiation.
        T0 = np.where(self.chromo, T_chr, T_cor)
        # gravitationally stratified density (hydrostatic scale height ~ kT/mg)
        n0 = np.where(self.chromo, n_chr, n_cor)
        self.rho = n0 * _MP
        self.v = np.zeros(n_cells)
        p = 2.0 * n0 * _KB * T0                      # p = (n_e+n_H) kB T, n_e=n_H
        self.E = p / (self.gamma - 1.0) + 0.5 * self.rho * self.v**2

        # frozen reservoir state at the very footpoints (Dirichlet mass reservoir)
        self._rho_res = self.rho.copy()
        self._E_res = self.E.copy()
        self.reservoir = (self.s < 0.4 * s_chr) | (self.s > L_loop - 0.4 * s_chr)

        self.t = 0.0

    # -- thermodynamic helpers ----------------------------------------------
    def primitives(self):
        rho = np.maximum(self.rho, 1e-30)
        v = self.v
        p = (self.gamma - 1.0) * (self.E - 0.5 * rho * v**2)
        p = np.maximum(p, 1e-6)
        n = rho / _MP                                # number density (=n_e=n_H)
        T = p / (2.0 * n * _KB)
        return rho, v, p, n, T

    def _flux(self, rho, v, p, E):
        return np.array([rho * v,
                         rho * v**2 + p,
                         (E + p) * v])

    # -- one hydro step (Rusanov / local Lax-Friedrichs, robust & positive) --
    def _hydro_step(self, dt):
        rho, v, p, n, T = self.primitives()
        U = np.array([rho, rho * v, self.E])
        F = self._flux(rho, v, p, self.E)
        cs = np.sqrt(self.gamma * p / rho)
        smax = np.abs(v) + cs

        # interface values with periodic-safe reflecting handling: use one-sided
        # (first order) Rusanov flux at cell faces i+1/2 between i and i+1
        UL, UR = U[:, :-1], U[:, 1:]
        FL, FR = F[:, :-1], F[:, 1:]
        a = np.maximum(smax[:-1], smax[1:])
        Fhalf = 0.5 * (FL + FR) - 0.5 * a * (UR - UL)   # faces 1..N-1

        dUdt = np.zeros_like(U)
        dUdt[:, 1:-1] = -(Fhalf[:, 1:] - Fhalf[:, :-1]) / self.ds
        # reflecting walls at both footpoints (closed loop): zero mass/energy
        # flux, momentum flux = pressure only
        # left wall (face at 0.5): mirror
        Fwall_L = np.array([0.0, p[0], 0.0])
        Fwall_R = np.array([0.0, p[-1], 0.0])
        dUdt[:, 0] = -(Fhalf[:, 0] - Fwall_L) / self.ds
        dUdt[:, -1] = -(Fwall_R - Fhalf[:, -1]) / self.ds

        Un = U + dt * dUdt
        self.rho = np.maximum(Un[0], 1e-24)
        self.v = Un[1] / self.rho
        self.E = Un[2]
        # energy floor (avoid negative pressure)
        rho, v, p, n, T = self.primitives()
        self.E = p / (self.gamma - 1.0) + 0.5 * self.rho * self.v**2

    # -- source terms: gravity, heating, radiation --------------------------
    def _sources(self, dt, Q):
        rho, v, p, n, T = self.primitives()
        # gravity (momentum + kinetic energy)
        mom = rho * v + dt * rho * self.g_par
        self.v = mom / np.maximum(rho, 1e-30)
        # heating and radiation act on internal energy
        Rrad = n * n * radiative_loss(T)             # W/m^3, n_e=n_H=n
        # limit radiative loss to a fraction of internal energy per step
        u = p / (self.gamma - 1.0)
        dU_rad = np.minimum(Rrad * dt, 0.5 * u)
        du = (Q - dU_rad / dt) * dt                  # net internal-energy change
        # add gravity work using updated v
        u_new = np.maximum(u + du, 1e-6)
        self.E = u_new + 0.5 * rho * self.v**2

    # -- implicit Spitzer conduction (unconditionally stable) ---------------
    def _conduction_step(self, dt):
        rho, v, p, n, T = self.primitives()
        C = 3.0 * n * _KB                            # volumetric heat capacity du/dT
        ds = self.ds
        # face conductivities kappa0 T^{5/2}
        Tf = 0.5 * (T[:-1] + T[1:])
        kf = _KAPPA0 * Tf**2.5                       # faces 1..N-1
        # build tridiagonal  (C/dt) T^{n+1} - d/ds(k dT/ds) = (C/dt) T^n
        N = self.N
        lower = np.zeros(N)
        diag = np.zeros(N)
        upper = np.zeros(N)
        rhs = C / dt * T
        # interior + insulated ends (zero conductive flux through footpoint walls)
        kL = np.zeros(N)
        kR = np.zeros(N)
        kR[:-1] = kf
        kL[1:] = kf
        diag = C / dt + (kL + kR) / ds**2
        lower[1:] = -kL[1:] / ds**2
        upper[:-1] = -kR[:-1] / ds**2
        Tnew = _thomas(lower, diag, upper, rhs)
        Tnew = np.maximum(Tnew, 3.0e3)
        # update energy with the temperature change (conserves energy in flux form)
        du = C * (Tnew - T)
        self.E = self.E + du

    def _apply_reservoir(self):
        # hold the deep-chromosphere footpoints at their initial cool dense state
        self.rho[self.reservoir] = self._rho_res[self.reservoir]
        self.E[self.reservoir] = self._E_res[self.reservoir]
        self.v[self.reservoir] = 0.0

    def step(self, dt, Q):
        self._hydro_step(dt)
        self._sources(dt, Q)
        self._conduction_step(dt)
        self._apply_reservoir()
        self.t += dt

    def cfl_dt(self, cfl=0.3):
        rho, v, p, n, T = self.primitives()
        cs = np.sqrt(self.gamma * p / rho)
        return cfl * self.ds / np.max(np.abs(v) + cs)


class HeatingDriver:
    r"""Build Q(s,t) for the loop by sampling the plasmoid-chain heating.

    The local reconnection heating (eta*j^2 + nu*omega^2) is integrated across
    the current sheet at each x to give a heating-per-column profile, mapped onto
    the coronal part of the loop, and replayed as a repeating train of impulsive
    episodes -- a nanoflare train [Parker 1988 [L8]; Cozzo et al. 2026 [L2]].

    Amplitude note (as in the 'proof of principle' of Reid et al. 2021 [L1]): the
    idealised MHD run uses an artificially small Lundquist number, so its Ohmic
    heating is not in coronal physical units.  We therefore keep the SPATIAL
    pattern (multiple heating sites from the chain) and TEMPORAL burstiness from
    the MHD, and calibrate the coronal-mean amplitude to `Q_mean`, a
    nanoflare-appropriate volumetric rate that sustains a ~1 MK loop.
    """

    def __init__(self, sim, frames, loop, s_per_tauA=4.0, Q_mean=3.0e-5,
                 band_a=3.0):
        L, a = sim.cfg.L, sim.cfg.a
        eta, nu, dx = sim.cfg.eta, sim.cfg.nu, sim.dx
        yband = (np.abs(sim.y - 0.25 * L) < band_a * a)   # around one sheet
        xgrid = sim.x[:, 0]

        prof, ts = [], []
        for fr in frames:
            H = eta * fr["j"]**2 + nu * fr["w"]**2        # heating rate (code)
            col = (H * yband).sum(axis=1) * dx            # integrate across sheet
            prof.append(col)
            ts.append(fr["t"])
        prof = np.array(prof)
        ts = np.array(ts)

        # map x in [0,L] onto the coronal cells of the loop
        scor = loop.s[loop.corona]
        xq = (scor - scor.min()) / (scor.max() - scor.min()) * L
        Qcor = np.array([np.interp(xq, xgrid, prof[k], period=L)
                         for k in range(len(frames))])
        m = Qcor.mean()
        if m > 0:
            Qcor *= Q_mean / m                            # calibrate amplitude

        self.Qfull = np.zeros((len(frames), loop.N))
        self.Qfull[:, loop.corona] = Qcor
        self.ts_phys = (ts - ts[0]) * s_per_tauA          # seconds within one episode
        dtf = np.diff(self.ts_phys).mean() if len(ts) > 1 else 1.0
        self.T_ep = self.ts_phys[-1] + dtf                # episode duration

    def Q_of(self, t):
        te = t % self.T_ep
        k = int(np.clip(np.searchsorted(self.ts_phys, te) - 1,
                        0, self.Qfull.shape[0] - 1))
        return self.Qfull[k]


def run_loop(loop, driver, T_loop, dt_max=2.0, record_dt=20.0):
    """Evolve the 1D loop under the reconnection-driven heating train."""
    recs = {"t": [], "T": [], "n": [], "v": [], "Qc": []}
    apex = loop.N // 2
    next_rec = 0.0
    while loop.t < T_loop:
        dt = min(loop.cfl_dt(0.3), dt_max)
        Q = driver.Q_of(loop.t)
        loop.step(dt, Q)
        if loop.t >= next_rec:
            rho, v, p, n, T = loop.primitives()
            recs["t"].append(loop.t)
            recs["T"].append(T.copy())
            recs["n"].append(n.copy())
            recs["v"].append(v.copy())
            recs["Qc"].append(Q.copy())
            next_rec += record_dt
        if not np.isfinite(loop.E).all():
            print("  [loop] non-finite state at t=%.1f" % loop.t)
            break
    for k in recs:
        recs[k] = np.array(recs[k])
    return recs


def _thomas(a, b, c, d):
    """Solve tridiagonal system (a=sub, b=diag, c=super, d=rhs)."""
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def run_mhd(N, S, a, t_end, n_frames, quiet=True, seed_modes=24, seed_amp=6e-3):
    """Run a plasmoid-chain MHD simulation and return (sim, frames)."""
    import io, contextlib
    cfg = PlasmoidConfig()
    cfg.N, cfg.eta, cfg.nu = N, 1.0 / S, 1.0 / S
    cfg.a, cfg.t_end, cfg.n_frames = a, t_end, n_frames
    cfg.seed_modes, cfg.seed_amp = seed_modes, seed_amp
    sim = PlasmoidMHD(cfg)
    frames = []

    def cap(s, n):
        ff = s.fields()
        ff["t"] = s._t
        frames.append(ff)

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(on_frame=cap)
    else:
        sim.run(on_frame=cap)
    return sim, frames


def diagnostics_timeseries(sim, frames, t_min_frac=0.2):
    """Compute X/O counts and reconnection diagnostics for every frame."""
    ts = np.array([fr["t"] for fr in frames])
    nX, nO, Mrate, Phidot = [], [], [], []
    Ider, Ish, ratio = [], [], []
    for fr in frames:
        cp = critical_points(sim, fr["psi"])
        rd = reconnection_diagnostics(sim, fr, cp)
        nX.append(cp["n_X"]); nO.append(cp["n_O"])
        Mrate.append(rd["M"]); Phidot.append(rd["Phi_dot"])
        Ider.append(rd["I_derived"]); Ish.append(rd["I_sheet"])
        ratio.append(rd["ratio"])
    out = dict(t=ts, nX=np.array(nX), nO=np.array(nO), M=np.array(Mrate),
               Phidot=np.array(Phidot), I_derived=np.array(Ider),
               I_sheet=np.array(Ish), ratio=np.array(ratio),
               M_SP=np.sqrt(sim.cfg.eta))          # Sweet-Parker rate = 1/sqrt(S)
    # DEVELOPED-chain frame: pick max island count in the nonlinear window,
    # excluding both the early broadband-seed transient and the late coalesced
    # state, so the hero panel shows a fully-formed (not seed-ripple) chain.
    t_end = sim.cfg.t_end
    win = np.where((ts > 0.4 * t_end) & (ts < 0.85 * t_end))[0]
    if win.size == 0:
        win = np.where(ts > t_min_frac * t_end)[0]
    out["peak"] = int(win[np.argmax(out["nO"][win])])
    return out


# ============================================================================
# 5.  Production: S-scan + hero chain + loop coupling -> one figure
# ============================================================================
def main(argv=None):
    argv = argv or sys.argv[1:]
    quick = "--quick" in argv

    print("\n>>> Plasmoid-chain reconnection + field-line-coupled coronal loop\n")
    params = coronal_parameters(verbose=True)
    tau_A = params.get("tau_A_s", 0.29)

    if quick:
        scan_S = [500, 2000, 8000]
        N_scan, N_hero = 128, 192
        a, t_end, nfr = 0.08, 12.0, 30
        S_hero = 8000
    else:
        scan_S = [500, 1500, 4000, 10000]
        N_scan, N_hero = 256, 384
        a, t_end, nfr = 0.055, 16.0, 60
        S_hero = 10000

    # ---- (A) S-scan: island count vs Lundquist number --------------------
    print("\n  [A] Lundquist-number scan (island count vs S) ...")
    scan = {"S": [], "nO": [], "nX": []}
    for S in scan_S:
        sim_s, frames_s = run_mhd(N_scan, S, a, t_end, 24)
        dts = diagnostics_timeseries(sim_s, frames_s)
        pk = dts["peak"]
        scan["S"].append(S)
        scan["nO"].append(int(dts["nO"][pk]))
        scan["nX"].append(int(dts["nX"][pk]))
        print(f"      S = {S:6d}  ->  peak islands N_O = {scan['nO'][-1]:2d}  "
              f"(N_X = {scan['nX'][-1]:2d})  at t = {dts['t'][pk]:.1f}")
    scan = {k: np.array(v) for k, v in scan.items()}
    # first S giving a chain (>= 3 islands)
    chain_mask = scan["nO"] >= 3
    S_first = int(scan["S"][chain_mask][0]) if chain_mask.any() else None
    print(f"      first plasmoid chain (N_O>=3) at S ~ {S_first}")

    # ---- (B) hero high-S run + diagnostic time series --------------------
    print(f"\n  [B] hero run: N={N_hero}, S={S_hero} ...")
    sim, frames = run_mhd(N_hero, S_hero, a, t_end, nfr, quiet=False)
    dts = diagnostics_timeseries(sim, frames)
    pk = dts["peak"]
    fpk = frames[pk]
    cp = critical_points(sim, fpk["psi"])
    rd = reconnection_diagnostics(sim, fpk, cp)
    print(f"      peak chain at t={fpk['t']:.1f}: N_O={cp['n_O']} islands, "
          f"N_X={cp['n_X']} X-lines; M={rd['M']:.4f}; "
          f"P=I*Phidot ratio={rd['ratio']:.2f}")

    # ---- (C) field-line coupling to the 1D loop --------------------------
    print("\n  [C] driving the 1D field-aligned loop with chain heating ...")
    loop = Loop1D(L_loop=50.0e6, n_cells=400, T_cor=0.8e6, n_cor=1.0e15)
    driver = HeatingDriver(sim, frames, loop, s_per_tauA=5.0, Q_mean=4.0e-5)
    T_loop = 1500.0 if quick else 2500.0
    recs = run_loop(loop, driver, T_loop=T_loop, record_dt=20.0)
    apex = loop.N // 2
    print(f"      loop sustained at T_apex = {recs['T'][:,apex].min()/1e6:.2f}"
          f"-{recs['T'][:,apex].max()/1e6:.2f} MK, "
          f"n_apex ~ {np.median(recs['n'][:,apex]):.2e} m^-3")

    # ---- (D) assemble the single figure ----------------------------------
    print("\n  building the single figure ...")
    make_figure(sim, frames, dts, cp, rd, scan, S_first, loop, recs, params,
                fname="plasmoid_chain_coronal_loop.png")
    print("\n  done.\n")


def make_figure(sim, frames, dts, cp, rd, scan, S_first, loop, recs, params,
                fname="plasmoid_chain_coronal_loop.png"):
    import matplotlib.gridspec as gridspec
    fpk = frames[dts["peak"]]
    L = sim.cfg.L
    tau_A = params.get("tau_A_s", 0.29)

    fig = plt.figure(figsize=(16, 15))
    fig.patch.set_facecolor("#0a0a14")
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.15, 1.0, 1.0],
                           hspace=0.36, wspace=0.30,
                           left=0.06, right=0.965, top=0.925, bottom=0.055)

    def darkax(ax, title=None):
        ax.set_facecolor("#12121e")
        ax.tick_params(colors="w", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#444")
        if title:
            ax.set_title(title, color="w", fontsize=10.5)

    fig.suptitle(
        "Plasmoid-chain magnetic reconnection & field-line-coupled coronal loop\n"
        f"S = {int(1/sim.cfg.eta)},  {sim.cfg.N}x{sim.cfg.N} spectral   |   "
        f"a current sheet fragmenting into a chain of magnetic islands",
        color="w", fontsize=15, fontweight="bold")

    # ---- HERO: plasmoid chain (current density + field lines + X/O) ----
    axh = fig.add_subplot(gs[0, :])
    j = fpk["j"]
    jm = np.percentile(np.abs(j), 99.6)
    im = axh.imshow(j.T, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r",
                    vmin=-jm, vmax=jm, aspect="auto")
    axh.contour(sim.x, sim.y, fpk["psi"], levels=70, colors="k",
                linewidths=0.3, alpha=0.5)
    axh.plot(cp["x_X"], cp["y_X"], "X", color="#39ff14", ms=11, mew=1.5,
             mec="k", ls="none", label=f"X-lines (reconnection sites): {cp['n_X']}")
    axh.plot(cp["x_O"], cp["y_O"], "o", color="#00e5ff", ms=8, mfc="none",
             mew=2.2, ls="none", label=f"O-points (plasmoids/islands): {cp['n_O']}")
    axh.set_title(
        f"Current density j & magnetic field lines at t = {fpk['t']:.1f} "
        r"$\tau_A$" f"  ({fpk['t']*tau_A:.1f} s):  the sheet has torn into "
        f"{cp['n_O']} islands", color="w", fontsize=11.5)
    axh.set_xticks([]); axh.set_yticks([])
    for s in axh.spines.values():
        s.set_color("#444")
    lg = axh.legend(loc="upper right", fontsize=9, facecolor="#0a0a14",
                    edgecolor="#444", labelcolor="w")
    cb = fig.colorbar(im, ax=axh, fraction=0.025, pad=0.01)
    cb.ax.tick_params(colors="w", labelsize=7); cb.outline.set_edgecolor("#444")

    # ---- (1) X-line / O-point count vs time ----
    ax1 = fig.add_subplot(gs[1, 0]); darkax(ax1, "X-line & island count vs time")
    ax1.plot(dts["t"], dts["nX"], color="#39ff14", lw=2, label="N$_X$ (X-lines)")
    ax1.plot(dts["t"], dts["nO"], color="#00e5ff", lw=2, label="N$_O$ (islands)")
    ax1.axvline(fpk["t"], color="w", ls=":", lw=1, alpha=0.6)
    ax1.set_xlabel(r"t / $\tau_A$", color="w", fontsize=9)
    ax1.set_ylabel("count", color="w", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#12121e", edgecolor="#444", labelcolor="w")
    ax1.grid(alpha=0.15, color="w")

    # ---- (2) S-scan: N_islands vs Lundquist number ----
    ax2 = fig.add_subplot(gs[1, 1])
    darkax(ax2, "chain onset: islands vs Lundquist number S")
    ax2.plot(scan["S"], scan["nO"], "o-", color="#ffcc00", lw=2, ms=8,
             label="peak N$_O$")
    Sref = np.array(scan["S"], float)
    ax2.plot(Sref, scan["nO"].max() * (Sref / Sref.max())**0.375, "--",
             color="#ff5555", lw=1.5, label=r"$\propto S^{3/8}$ [T4]")
    if S_first:
        ax2.axvline(S_first, color="#39ff14", ls=":", lw=1.5,
                    label=f"first chain: S$\\approx${S_first}")
    ax2.set_xscale("log"); ax2.set_xlabel("S = 1/η", color="w", fontsize=9)
    ax2.set_ylabel("peak island count", color="w", fontsize=9)
    ax2.legend(fontsize=7.5, facecolor="#12121e", edgecolor="#444", labelcolor="w")
    ax2.grid(alpha=0.15, color="w", which="both")

    # ---- (3) reconnection rate M(t) ----
    ax3 = fig.add_subplot(gs[1, 2]); darkax(ax3, "reconnection rate  M = E$_{rec}$/(v$_A$B)")
    ax3.plot(dts["t"], dts["M"], color="#ff9933", lw=2, label="measured M(t)")
    ax3.axhline(dts["M_SP"], color="#5599ff", ls="--", lw=1.5,
                label=r"Sweet-Parker $S^{-1/2}$")
    ax3.axhline(0.01, color="#aa66ff", ls=":", lw=1.5,
                label="fast (plasmoid) ~0.01")
    ax3.set_xlabel(r"t / $\tau_A$", color="w", fontsize=9)
    ax3.set_ylabel("M", color="w", fontsize=9)
    ax3.set_yscale("log")
    ax3.legend(fontsize=7.5, facecolor="#12121e", edgecolor="#444", labelcolor="w")
    ax3.grid(alpha=0.15, color="w", which="both")

    # ---- (4) explicit P = I * Phidot test ----
    ax4 = fig.add_subplot(gs[2, 0]); darkax(ax4, r"Longcope-Tarr test:  I = P/$\dot\Phi$")
    good = dts["I_sheet"] > 0
    ax4.plot(dts["t"][good], dts["I_sheet"][good], color="#00e5ff", lw=2,
             label=r"I$_{sheet}=\int|j|\,dA$")
    ax4.plot(dts["t"][good], dts["I_derived"][good], color="#ffcc00", lw=2, ls="--",
             label=r"I = P$_{ohm}$/$\dot\Phi$")
    ax4.set_xlabel(r"t / $\tau_A$", color="w", fontsize=9)
    ax4.set_ylabel("current (code units)", color="w", fontsize=9)
    med = np.median(dts["ratio"][np.isfinite(dts["ratio"]) & (dts["ratio"] > 0)])
    ax4.text(0.05, 0.9, f"P = I·$\\dot\\Phi$  →  ratio ≈ {med:.1f}",
             transform=ax4.transAxes, color="w", fontsize=8.5)
    ax4.legend(fontsize=7.5, facecolor="#12121e", edgecolor="#444", labelcolor="w")
    ax4.grid(alpha=0.15, color="w")

    # ---- (5) 1D loop temperature map T(s,t) ----
    ax5 = fig.add_subplot(gs[2, 1]); darkax(ax5, "coupled 1D loop:  log$_{10}$ T(s,t)")
    ext = [0, loop.L / 1e6, recs["t"][0], recs["t"][-1]]
    imt = ax5.imshow(np.log10(recs["T"]), origin="lower", aspect="auto",
                     extent=ext, cmap="inferno", vmin=4, vmax=6.3)
    ax5.set_xlabel("s along loop [Mm]", color="w", fontsize=9)
    ax5.set_ylabel("t [s]", color="w", fontsize=9)
    cb5 = fig.colorbar(imt, ax=ax5, fraction=0.046, pad=0.02)
    cb5.ax.tick_params(colors="w", labelsize=7); cb5.outline.set_edgecolor("#444")

    # ---- (6) loop apex T & n vs time ----
    ax6 = fig.add_subplot(gs[2, 2]); darkax(ax6, "loop apex response (sustained corona)")
    apex = loop.N // 2
    ax6.plot(recs["t"], recs["T"][:, apex] / 1e6, color="#ff5555", lw=2,
             label="T$_{apex}$ [MK]")
    ax6.set_xlabel("t [s]", color="w", fontsize=9)
    ax6.set_ylabel("T [MK]", color="#ff5555", fontsize=9)
    ax6.tick_params(axis="y", colors="#ff5555")
    ax6b = ax6.twinx()
    ax6b.plot(recs["t"], recs["n"][:, apex], color="#55aaff", lw=1.6,
              label="n$_{apex}$")
    ax6b.set_ylabel("n [m$^{-3}$]", color="#55aaff", fontsize=9)
    ax6b.tick_params(axis="y", colors="#55aaff", labelsize=8)
    for s in ax6b.spines.values():
        s.set_color("#444")

    fig.savefig(fname, dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {fname}")


if __name__ == "__main__":
    main()
