#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Parametric scaling study of the 2D resistive-MHD reconnection / plasmoid code
=============================================================================

Question: does the code reproduce the *trends* that reconnection theory predicts
when we vary the Lundquist number S = 1/eta (and sheet thickness a, resolution N)?
Each test below has a theoretical prediction that is written down BEFORE looking
at the code output; the verdict (match / partial / mismatch) is reported by the
analysis stage, not assumed.

Tests
-----
 A. Linear tearing (verification + scaling).  The code is run in the linear regime
    with the equilibrium held fixed and a tiny multi-mode seed; the measured growth
    rates gamma(m) are compared with the eigenvalues of the *same* linearised
    equations (tearing_theory.py).  The eigensolver is then used to extract the
    scalings  gamma_max ~ S_a^{-1/2},  k_max a ~ S_a^{-1/4}  (FKR/Coppi theory).
 B. Onset time.  t_onset ~ 1/gamma_max(S_a)  => t_onset grows with S (the exponent
    follows the *local* eigenvalue slope, not the asymptotic 1/2).
 C. Local Sweet-Parker structure of X-point current layers,
        delta / ell  ~  S_ell^{-1/2},    S_ell = ell * B_up / eta.
    RESULT: NOT testable in this configuration.  With the Harris sheet held fixed
    its thickness stays pinned at a (> delta_SP at these S), so delta/ell is flat in
    S_ell.  A probe of X-point collapse in a thick un-held sheet (S=2000, a=0.4,
    eps=0.03) thinned the layer only 1.6x (j_X: 2.3 -> 3.8, Sweet-Parker would need
    ~14), i.e. the collapse regime was not reached.  A real Sweet-Parker test needs a
    larger seed and N >~ 384; the Sweet-Parker rate scaling is therefore reported as
    NOT demonstrated, rather than inferred.
 D. Island (plasmoid) number vs S for a *self-selected* chain (tiny random seed):
        N_islands ~ 2 k_max L / 2 pi  (linear theory, two sheets).  The S^{3/8}
        plasmoid-regime law (Samtaney et al. 2009) needs S_L >~ 1e4 sheets and is
        NOT reached here.
 E. Resolution convergence of the nonlinear runs.
 F. Energy-budget closure (E_B + E_K + Q_ohm + Q_visc - W_forcing conserved).
    The Longcope-Tarr P ~ I*Phi_dot comparison of the previous study is dropped: a
    held sheet satisfies P = I*E_0 trivially (E_0 = eta*j0 is the external field), so
    the earlier 'ratio ~ 1-2.7' was not a discriminating test.

IMPORTANT CORRECTION to plasmoid_chain_coronal_loop.py: there the sheet was NOT held, so
it diffused on t ~ a^2/eta (1.5 tau_A at S=500, ~30 tau_A at S=1e4) -- comparable to the
run length -- and the 'chain' was dominated by the islands imposed through the 24-mode
seed (N_O = 32-36 at t=0), not selected by tearing.  The island counts and 'first chain at
S~500' reported there should not be read as a demonstration of plasmoid-chain onset.

References
----------
 [FKR]   Furth, Killeen & Rosenbluth 1963, Phys. Fluids 6, 459.
 [Coppi] Coppi, Galvao, Pellat, Rosenbluth & Rutherford 1976, Sov. J. Plasma Phys. 2, 533.
 [SP]    Parker 1957, J. Geophys. Res. 62, 509; Sweet 1958, IAU Symp. 6, 123.
 [L07]   Loureiro, Schekochihin & Cowley 2007, Phys. Plasmas 14, 100703.
 [S09]   Samtaney, Loureiro, Uzdensky, Schekochihin & Cowley 2009, PRL 103, 105004.
 [B09]   Bhattacharjee, Huang, Yang & Rogers 2009, Phys. Plasmas 16, 112102.
 [U10]   Uzdensky, Loureiro & Schekochihin 2010, PRL 105, 235002.
 [Ser10] Servidio et al. 2010, Phys. Plasmas 17, 032315.
 [LT15]  Longcope & Tarr 2015, Phil. Trans. R. Soc. A 373, 20140263.

Usage:  python scaling_study_2d.py theory | linear | nonlinear | converge | all
Results are cached in ./scaling_results/ so the analysis can be re-run cheaply.
"""
from __future__ import annotations

import os
os.environ.setdefault("MHD_FFT_WORKERS", "4")      # per-process FFT threads

import sys
import io
import json
import time
import contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "scaling_results")
os.makedirs(OUT, exist_ok=True)

import mhd_reconnection_coronal_heating as base          # noqa: E402
import plasmoid_chain_coronal_loop as P                   # noqa: E402
import tearing_theory as T                                # noqa: E402


# ============================================================================
# A.  Linear tearing benchmark: code vs eigenvalue solver
# ============================================================================
class LinearCfg(base.SimulationConfig):
    N = 192
    L = 2.0 * np.pi
    a = 0.15
    eta = 4e-4
    nu = 4e-4
    kappa = 1e-4
    vpert = 0.0
    cfl = 0.3
    dt_max = 1.0e-2
    t_end = 60.0
    n_frames = 240
    mode_list = (1, 2, 3, 4, 5, 6)
    seed_amp = 1.0e-6
    seed_rng = 7
    hold = True


class LinearTearingMHD(P.PlasmoidMHD):
    """Solver with the Harris equilibrium *held fixed* (external E-field balances
    its resistive diffusion) so linear growth rates are not contaminated by the
    sheet broadening; the seed is a tiny multi-mode perturbation."""

    def _initial_condition(self):
        cfg = self.cfg
        L, a = cfg.L, cfg.a
        x, y = self.x, self.y
        Bx = np.tanh((y - 0.25 * L) / a) - np.tanh((y - 0.75 * L) / a)
        Bx -= Bx.mean()
        Bx_hat = self._fft(Bx)
        psi_hat = np.zeros_like(Bx_hat)
        nz = self.ky != 0
        psi_hat[nz] = -Bx_hat[nz] / (1j * self.ky[nz])
        self.psi0_hat = psi_hat.copy()
        rng = np.random.default_rng(cfg.seed_rng)
        g = (np.exp(-((y - 0.25 * L) / (2 * a))**2)
             + np.exp(-((y - 0.75 * L) / (2 * a))**2))
        d = np.zeros_like(y)
        for m in cfg.mode_list:
            d += np.cos(m * 2.0 * np.pi / L * x + rng.uniform(0, 2 * np.pi))
        d *= cfg.seed_amp * g
        psi_hat = psi_hat + self._fft(d) * self.dealias
        self._hold_forcing = cfg.eta * self.k2 * self.psi0_hat
        return psi_hat, np.zeros_like(psi_hat), self._fft(np.ones_like(y))

    def _rhs(self, psi_hat, w_hat, T_hat):
        r = super()._rhs(psi_hat, w_hat, T_hat)
        if getattr(self.cfg, "hold", False):
            return (r[0] + self._hold_forcing, r[1], r[2])
        return r


def mode_amplitudes(sim, modes):
    N = sim.cfg.N
    return np.array([np.sqrt(np.sum(np.abs(sim.psi_hat[m, :])**2)) / N**2
                     for m in modes])


def fit_gamma(t, A, A_hi=1.0e-4, efolds=4.0, min_pts=8):
    """Exponential growth rate from the last `efolds` e-folds below A_hi."""
    lnA = np.log(np.maximum(A, 1e-300))
    above = np.where(A > A_hi)[0]
    end = above[0] if above.size else len(A) - 1
    if end < min_pts:
        return np.nan, "short"
    below = np.where(lnA[:end + 1] < lnA[end] - efolds)[0]
    if below.size:
        i0 = below[-1] + 1
        flag = "ok"
    else:                                   # not enough dynamic range
        i0 = int(0.6 * end)
        flag = "range"
    sel = slice(i0, end + 1)
    if end + 1 - i0 < min_pts:
        return np.nan, "short"
    c = np.polyfit(t[sel], lnA[sel], 1)
    return float(c[0]), flag


def linear_job(eta, N, a, t_end, modes=(1, 2, 3, 4, 5, 6), seed_amp=1e-6):
    cfg = LinearCfg()
    cfg.N, cfg.eta, cfg.nu, cfg.a = N, eta, eta, a
    cfg.t_end, cfg.mode_list, cfg.seed_amp = t_end, tuple(modes), seed_amp
    cfg.n_frames = int(t_end / 0.25)
    sim = LinearTearingMHD(cfg)
    ts, amps = [], []

    def cb(s, n):
        ts.append(s._t)
        amps.append(mode_amplitudes(s, modes))

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(on_frame=cb)
    ts, amps = np.array(ts), np.array(amps)
    gam, flags = [], []
    for k, m in enumerate(modes):
        g, fl = fit_gamma(ts, amps[:, k])
        gam.append(g)
        flags.append(fl)
    return dict(eta=eta, N=N, a=a, modes=list(modes), gamma=gam, flags=flags,
                t=ts.tolist(), amps=amps.tolist(), wall=time.time() - t0)


def stage_linear(workers=3):
    from concurrent.futures import ProcessPoolExecutor
    a = 0.15
    # (eta, N, t_end): t_end ~ 11 e-folds / gamma_max (estimated from the eigensolver)
    jobs = [(1.0e-3, 128, 60.0), (4.0e-4, 192, 90.0), (1.5e-4, 256, 130.0),
            (6.0e-5, 256, 190.0), (6.0e-5, 192, 190.0)]
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(linear_job, e, n, a, te) for (e, n, te) in jobs]
        for f, (e, n, te) in zip(futs, jobs):
            r = f.result()
            print("  linear eta=%.1e N=%d: gamma(m)=%s  [%.0fs]" %
                  (e, n, np.round(r["gamma"], 4), r["wall"]), flush=True)
            res.append(r)
    json.dump(res, open(os.path.join(OUT, "linear_code.json"), "w"))
    return res


# ---- eigen-theory side -----------------------------------------------------
def theory_job(Sa, a=0.15):
    """gamma(k) on a log grid in ka for sheet Reynolds number S_a (periodic solver)."""
    eta = a / Sa
    Ny = 256 if Sa <= 300 else 512 if Sa <= 1500 else 768 if Sa <= 4000 else 1024
    kas = np.array([0.04, 0.06, 0.08, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4, 0.55, 0.75])
    g = np.array([T.growth_rate(ka / a, eta, eta, a, Ny=Ny) for ka in kas])
    i = int(np.argmax(g))
    # refine around the maximum
    lo, hi = kas[max(i - 1, 0)], kas[min(i + 1, len(kas) - 1)]
    kr = np.exp(np.linspace(np.log(lo), np.log(hi), 5))
    gr = np.array([T.growth_rate(ka / a, eta, eta, a, Ny=Ny) for ka in kr])
    kk = np.concatenate([kas, kr])
    gg = np.concatenate([g, gr])
    o = np.argsort(kk)
    kk, gg = kk[o], gg[o]
    j = int(np.argmax(gg))
    if 0 < j < len(kk) - 1:
        x = np.log(kk[j - 1:j + 2])
        c = np.polyfit(x, gg[j - 1:j + 2], 2)
        if c[0] < 0:
            xm = -c[1] / (2 * c[0])
            kmax, gmax = float(np.exp(xm)), float(np.polyval(c, xm))
        else:
            kmax, gmax = float(kk[j]), float(gg[j])
    else:
        kmax, gmax = float(kk[j]), float(gg[j])
    return dict(Sa=Sa, a=a, Ny=Ny, ka=kk.tolist(), gamma=gg.tolist(),
                gmax=gmax, kamax=kmax)


def stage_theory(workers=3):
    from concurrent.futures import ProcessPoolExecutor
    Sas = [30, 100, 300, 1000, 3000, 6000]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(theory_job, Sas))
    for r in res:
        print("  theory Sa=%6.0f: gamma_max*a=%.5f  ka_max=%.3f  (Ny=%d)" %
              (r["Sa"], r["gmax"] * r["a"], r["kamax"], r["Ny"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "theory_eigen.json"), "w"))
    return res


# ============================================================================
# B-F.  Nonlinear scan diagnostics
# ============================================================================
def psi_rec(sim, psi):
    """Reconnected flux per sheet: 1/2 * integral |d psi/dx| dx along the neutral
    line of each sheet = sum over islands of (psi_O - psi_X)  (flux-transfer
    measure that does not depend on the electrostatic/gauge choice)."""
    N = sim.cfg.N
    px = sim._ifft(sim._ddx(sim._fft(psi)))
    vals = [0.5 * np.sum(np.abs(px[:, jj])) * sim.dx for jj in (N // 4, 3 * N // 4)]
    return float(np.mean(vals))


def sheet_budget(sim, f, a, halfwidth_a=10.0):
    """Ohmic power and signed current in the two sheet bands."""
    L = sim.cfg.L
    eta = sim.cfg.eta
    dA = sim.dx**2
    j = f["j"]
    out = []
    for ys in (0.25 * L, 0.75 * L):
        band = np.abs(sim.y - ys) < halfwidth_a * a
        out.append((eta * np.sum(j[band]**2) * dA, abs(np.sum(j[band]) * dA)))
    return out        # [(P_ohm, I), (P_ohm, I)]


def local_sheet_samples(sim, f, cp, a, nmax=4):
    """Measure thickness delta, length ell and upstream field B_up of the current
    layers through the strongest X-points (for the local Sweet-Parker test)."""
    from scipy.ndimage import map_coordinates
    L, dx = sim.cfg.L, sim.dx
    j, Bx = f["j"], f["Bx"]
    samples = []
    if cp["n_X"] == 0 or cp["n_O"] < 2:
        return samples
    order = np.argsort(-cp["j_X"])[:nmax]
    ns = 801
    for i in order:
        xX, yX = cp["x_X"][i], cp["y_X"][i]
        same = np.abs(cp["y_O"] - yX) < 5 * a
        if same.sum() < 2:
            continue
        d = (cp["x_O"][same] - xX + L / 2) % L - L / 2
        r, l = d[d > 0], -d[d < 0]
        if r.size == 0 or l.size == 0:
            continue
        ell = r.min() + l.min()                      # O-point to O-point spacing
        span = 8 * a
        ys = yX + np.linspace(-span, span, ns)
        coords = [np.full(ns, xX / dx), (ys % L) / dx]
        jp = np.abs(map_coordinates(j, coords, order=1, mode="wrap"))
        ic = int(np.argmax(jp[ns // 2 - 60: ns // 2 + 61])) + ns // 2 - 60
        jpk = jp[ic]
        half = 0.5 * jpk
        up = np.where(jp[ic:] < half)[0]
        dn = np.where(jp[:ic][::-1] < half)[0]
        if up.size == 0 or dn.size == 0:
            continue
        dy = 2 * span / (ns - 1)
        delta = 0.5 * (up[0] + dn[0]) * dy
        bu = []
        for sgn in (-1, 1):
            yb = ys[ic] + sgn * 4.0 * delta
            bu.append(abs(map_coordinates(Bx, [[xX / dx], [(yb % L) / dx]],
                                          order=1, mode="wrap")[0]))
        Bup = float(np.mean(bu))
        samples.append(dict(delta=float(delta), ell=float(ell), Bup=Bup,
                            jpk=float(jpk), resolved=bool(delta > 2.0 * dx)))
    return samples


class NonlinCfg(LinearCfg):
    """Nonlinear tearing of a HELD Harris sheet (an external E-field sustains the
    current sheet against resistive diffusion, so the sheet thickness `a` -- and
    hence the sheet Lundquist number S_a = a v_A/eta -- stays fixed during the run).
    Without this the sheet diffuses on t ~ a^2/eta (1.5 tau_A at S=500; ~30 tau_A at
    S=1e4), faster than or comparable to the tearing time, and the 'chain' is then
    dominated by the islands imposed through the seed rather than selected by the
    instability."""
    hold = True
    cfl = 0.3
    dt_max = 6.0e-3


def nonlinear_job(S, N, a, t_end, seed_amp=1.0e-5, seed_modes=16, rng_seed=1,
                  n_frames=None, tag=""):
    cfg = NonlinCfg()
    cfg.N, cfg.eta, cfg.nu = N, 1.0 / S, 1.0 / S
    cfg.a, cfg.t_end = a, t_end
    cfg.mode_list = tuple(range(1, seed_modes + 1))
    cfg.seed_amp, cfg.seed_rng = seed_amp, rng_seed
    cfg.n_frames = n_frames or int(t_end / 0.2)
    sim = LinearTearingMHD(cfg)
    j0 = sim._ifft(-sim.k2 * sim.psi0_hat)             # held-equilibrium current

    rec = dict(t=[], nO=[], nX=[], psirec=[], P1=[], I1=[], P2=[], I2=[],
               MX=[], jmax=[], PF=[], samples=[])

    def cb(s, n):
        f = s.fields()
        # power injected by the external forcing that holds the sheet:  eta <j j0>
        rec["PF"].append(float(cfg.eta * np.mean(f["j"] * j0)))
        cp = P.critical_points(s, f["psi"])
        rd = P.reconnection_diagnostics(s, f, cp)
        sb = sheet_budget(s, f, a)
        rec["t"].append(s._t)
        rec["nO"].append(cp["n_O"])
        rec["nX"].append(cp["n_X"])
        rec["psirec"].append(psi_rec(s, f["psi"]))
        rec["P1"].append(sb[0][0]); rec["I1"].append(sb[0][1])
        rec["P2"].append(sb[1][0]); rec["I2"].append(sb[1][1])
        rec["MX"].append(rd["M"])
        rec["jmax"].append(float(np.abs(f["j"]).max()))
        for smp in local_sheet_samples(s, f, cp, a):
            smp["t"] = s._t
            rec["samples"].append(smp)

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(on_frame=cb)
    h = sim.hist
    out = {k: (np.array(v).tolist() if k != "samples" else v) for k, v in rec.items()}
    out.update(S=S, N=N, a=a, t_end=t_end, seed_amp=seed_amp, rng_seed=rng_seed,
               tag=tag, dx=sim.dx, eta=cfg.eta,
               hist={k: np.array(v).tolist() for k, v in h.items()},
               wall=time.time() - t0)
    return out


def stage_nonlinear(workers=3, quick=False):
    from concurrent.futures import ProcessPoolExecutor
    a = 0.055
    # (S, N, t_end)
    jobs = [(500, 192, 26.0), (1000, 256, 28.0), (2000, 256, 34.0),
            (4000, 384, 42.0), (8000, 384, 50.0)]
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(nonlinear_job, S, N, a, te, tag="scan") for (S, N, te) in jobs]
        for f, (S, N, te) in zip(futs, jobs):
            r = f.result()
            print("  nonlinear S=%d N=%d done [%.0fs]  final nO=%d" %
                  (S, N, r["wall"], r["nO"][-1]), flush=True)
            res.append(r)
            json.dump(res, open(os.path.join(OUT, "nonlinear_scan.json"), "w"))
    return res


def stage_converge(workers=3):
    from concurrent.futures import ProcessPoolExecutor
    a, S, te = 0.055, 2000, 34.0
    jobs = [(S, 128, a, te), (S, 192, a, te), (S, 256, a, te)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(nonlinear_job, S, N, aa, t, tag="conv") for (S, N, aa, t) in jobs]
        res = [f.result() for f in futs]
    for r in res:
        print("  converge N=%d [%.0fs]" % (r["N"], r["wall"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "convergence.json"), "w"))
    return res


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("theory", "all"):
        print("[A1] eigenvalue theory ..."); stage_theory()
    if stage in ("linear", "all"):
        print("[A2] linear benchmark of the code ..."); stage_linear()
    if stage in ("nonlinear", "all"):
        print("[B-F] nonlinear S-scan ..."); stage_nonlinear()
    if stage in ("converge", "all"):
        print("[E] convergence ..."); stage_converge()
    print("stage %s finished in %.1f min" % (stage, (time.time() - t0) / 60))
