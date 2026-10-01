#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis + single figure for the 3D line-tied kink model (reads scaling_results/)."""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("MHD_FFT_WORKERS", "4")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "scaling_results")
import kink_loop_3d as K
import plasmoid_chain_coronal_loop as P      # Loop1D (Reid-style 1D loop)


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------- growth rate
def growth_rate_from_run(r, lo=1.0e-3, hi=4.0e-2, jnl=2.6):
    """Amplitude growth rate gamma = (1/2) d ln E_k / dt over the weakly nonlinear ramp
    lo < E_k < hi with max|j| < jnl*j_eq, after the initial Alfvenic transient.
    NOTE: the kick's oscillating transient (E_k ~ 1e-4) contaminates the linear phase,
    so near threshold this estimate is good to ~30% only."""
    t, Ek = np.array(r["t"]), np.array(r["Ek"])
    jr = np.array(r["jmax"]) / r["jeq"]
    imin = int(np.argmin(Ek[: max(3, len(Ek) // 2)]))
    imax = int(np.argmax(Ek))                    # growth phase only (exclude the decay after the peak)
    sel = np.where((Ek > lo) & (Ek < hi) & (jr < jnl) & (np.arange(len(t)) > imin) &
                   (np.arange(len(t)) <= imax))[0]
    if sel.size < 5 or Ek[-1] < 4 * Ek[imin]:
        return np.nan, (np.nan, np.nan)
    c = np.polyfit(t[sel], 0.5 * np.log(Ek[sel]), 1)
    return float(c[0]), (float(t[sel[0]]), float(t[sel[-1]]))


def stable_flag(r):
    Ek = np.array(r["Ek"]); jr = np.array(r["jmax"]) / r["jeq"]
    return (Ek[-1] < 4 * Ek.min() + 1e-12) and jr.max() < 1.5


def time_to_nonlinear(r, level=2.0):
    jr = np.array(r["jmax"]) / r["jeq"]
    i = np.where(jr >= level)[0]
    return float(np.array(r["t"])[i[0]]) if i.size else np.nan


# --------------------------------------------------------------------- 3D drawing
def draw_lines(ax, psi, W, L, B0, rings, title, jcol=None):
    Bx, By = K.field_components(psi, W)
    x0, y0, c0 = [], [], []
    for r0, nth in rings:
        th = np.linspace(0, 2 * np.pi, nth, endpoint=False) + 0.3
        x0 += list(r0 * np.cos(th)); y0 += list(r0 * np.sin(th)); c0 += [r0] * nth
    path = K.trace_lines(Bx, By, W, L, B0, np.array(x0), np.array(y0))
    z = np.linspace(0, L, path.shape[0])
    cmap = plt.get_cmap("plasma")
    for m in range(path.shape[2]):
        ax.plot(path[:, 0, m], path[:, 1, m], z, color=cmap(c0[m] / 1.0), lw=0.9, alpha=0.9)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 3.2))
    ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=14, azim=-58)
    return path


def main():
    prod_path = os.environ.get("KINK_PROD", os.path.join(RES, "kink_production.npz"))
    out_png = os.environ.get("KINK_OUT", os.path.join(HERE, "kink_loop_3d.png"))
    scan3 = load("kink_scan_p3.json") or []
    scan1 = load("kink_scan.json") or []
    refine = load("kink_refine.json") or []
    refine2 = load("kink_refine2.json") or []
    have_prod = os.path.exists(prod_path)
    summary = {}

    fig = plt.figure(figsize=(21, 16))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 4, hspace=0.30, wspace=0.38, left=0.04, right=0.985,
                          top=0.925, bottom=0.05)
    fig.suptitle("3D line-tied kink instability of a twisted coronal loop with zero net current "
                 "(reduced MHD, spectral + staggered grid)  →  heating  →  1D field-aligned loop",
                 fontsize=15, fontweight="bold")

    # ================================================================== threshold / growth rates
    ax_thr = fig.add_subplot(gs[1, 2])
    std = [x for x in refine + refine2 if abs(x["job"].get("L", 20) - 20) < 1 and
           x["job"].get("N", 64) == 64 and x["job"].get("nu", 2e-4) == 2e-4 and
           x["job"].get("Nc", 32) == 32]
    pts = {}
    for r in scan3 + std:
        key = round(r["Phi0"] / np.pi, 3)
        if key in pts and pts[key]["t_last"] >= r["t"][-1]:
            continue                       # keep the longest run at each twist
        st = stable_flag(r)
        g, _ = growth_rate_from_run(r)
        pts[key] = dict(g=0.0 if st else g, st=st, t_last=r["t"][-1], tnl=time_to_nonlinear(r))
    if pts:
        keys = sorted(pts)
        px = np.array(keys); py = np.array([pts[k]["g"] for k in keys])
        stb = np.array([pts[k]["st"] for k in keys]); tnl = np.array([pts[k]["tnl"] for k in keys])
        okn = np.isfinite(tnl)
        ax_thr.plot(px[okn], 1.0 / tnl[okn], "o", color="#d62728", ms=9,
                    label=r"unstable: $1/t_{nl}$ ($t_{nl}$: time to $|j|=2j_{eq}$)")
        if stb.any():
            ax_thr.plot(px[stb], np.zeros(stb.sum()), "s", color="#1f77b4", ms=9, mfc="none", mew=2,
                        label="stable through t_end")
        summary["Phi0_over_pi"] = px.tolist(); summary["gamma_Ek_ramp_noisy"] = [float(v) for v in py]
        summary["t_nonlinear"] = [float(v) for v in tnl]
        summary["stable_at_longest_run"] = [float(k) for k in px[stb]]
        sel = (px <= 5.01) & np.isfinite(tnl)
        if sel.sum() >= 3:
            c = np.polyfit(px[sel], 1.0 / tnl[sel], 1)       # 1/t_nl ~ gamma -> 0 at threshold
            Phi_c = float(-c[1] / c[0])
            xx = np.linspace(Phi_c, 5.2, 20)
            ax_thr.plot(xx, np.polyval(c, xx), "-", color="#d62728", lw=1.2, alpha=.6)
            ax_thr.axvline(Phi_c, color="goldenrod", lw=3, alpha=.7,
                           label="Φ_c ≈ %.2fπ  (1/t_nl → 0 extrapolation)" % Phi_c)
            summary["Phi_c_extrapolated_pi"] = Phi_c
    ax_thr.axvline(2.5, color="gray", ls="--", lw=1.2,
                   label="Hood & Priest 1981: 2.5π (uniform twist, net current)")
    if scan1:
        top = max(r["Phi0"] for r in scan1) / np.pi
        ax_thr.annotate("p=1 profile (twist\nconcentrated at axis):\nstable to %.0fπ" % top,
                        xy=(0.03, 0.30), xycoords="axes fraction", ha="left", fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", ec="#888"))
        summary["p1_profile_stable_up_to_pi"] = float(top)
    ax_thr.set_xlabel(r"axis twist $\Phi_0 = L B_\theta/(rB_0)|_{r\to0}$  [units of π]")
    ax_thr.set_ylabel(r"$1/t_{nl}\ [\tau_A^{-1}]$  (rate-like)")
    ax_thr.set_title("(f) kink threshold from the onset rate 1/t$_{nl}$  (N=64, ν=2e-4)", fontsize=10)
    ax_thr.legend(fontsize=6.8, loc="upper left"); ax_thr.grid(alpha=.3)

    # ================================================================== robustness / scaling tests
    ax_sc = fig.add_subplot(gs[1, 3])
    conv = load("kink_conv3d.json") or []
    base6 = next((r for r in scan3 if abs(r["Phi0"] / np.pi - 6) < 1e-6), None)
    base7 = next((r for r in scan3 if abs(r["Phi0"] / np.pi - 7) < 1e-6), None)
    tests = []
    if base6 is not None:
        b6 = time_to_nonlinear(base6)
        for r in refine:
            j = r["job"]
            if abs(j["Phi0"] / np.pi - 6) > 1e-6:
                continue
            tn = time_to_nonlinear(r)
            if j.get("L", 20) == 40:
                tests.append(("L = 40\n(6π)", tn / b6, 2.0))
            elif j.get("N", 64) == 96:
                tests.append(("N=96,Nc=48\n(6π)", tn / b6, 1.0))
            elif j.get("nu", 2e-4) == 1e-4:
                tests.append(("ν = 1e-4\n(6π)", tn / b6, 1.0))
    if base7 is not None:
        b7 = time_to_nonlinear(base7)
        for r in conv:
            j = r["job"]
            tn = time_to_nonlinear(r)
            lab = "N=%d,Nc=%d" % (j.get("N", 64), j.get("Nc", 32))
            if "nu" in j:
                lab += "\nν=%g" % j["nu"]
            tests.append((lab + "\n(7π)", tn / b7 if np.isfinite(tn) else np.nan, 1.0))
    if tests:
        xs = np.arange(len(tests))
        vals = np.array([t[1] for t in tests]); exps = np.array([t[2] for t in tests])
        top = np.nanmax(vals) if np.isfinite(vals).any() else 2.0
        for i, (lab, v, e) in enumerate(tests):
            if np.isfinite(v):
                ax_sc.bar(i, v, color="#2ca02c" if abs(v - e) / e < 0.25 else "#ff7f0e", alpha=.85, width=.6)
                ax_sc.plot([i - .35, i + .35], [e, e], "k--", lw=1.4)
            else:
                ax_sc.text(i, 0.1 * top, "no growth\nby t_end", ha="center", fontsize=7, color="#1f77b4")
        ax_sc.set_xticks(xs)
        ax_sc.set_xticklabels([t[0].replace("\n", " ") for t in tests], fontsize=6.5,
                              rotation=35, ha="right")
        ax_sc.set_ylabel("t$_{nl}$ / t$_{nl}$(baseline N=64, Nc=32, ν=2e-4)")
        ax_sc.set_ylim(0, 1.25 * max(top, 2.2))
        summary["robustness_tnl_ratio"] = {t[0].replace("\n", " "): (None if not np.isfinite(t[1]) else float(t[1]))
                                           for t in tests}
    ax_sc.set_title("(g) scaling & numerical-robustness tests (dashed = expected)", fontsize=10)
    ax_sc.grid(alpha=.3, axis="y")

    if not have_prod:
        fig.savefig(out_png, dpi=100, facecolor="white")
        print("no production data yet; partial figure saved", summary)
        return

    D = np.load(prod_path, allow_pickle=True)
    meta = json.loads(str(D["meta"]))
    W, L, Nc, N = meta["W"], meta["L"], meta["Nc"], meta["N"]
    t_snap, psi_all, j_all, H_all = D["t_snap"], D["psi"], D["j"], D["H"]
    t = D["t"]; Em = D["Em"]; Ek = D["Ek"]; Qo = D["Qo"]; Qv = D["Qv"]; jmax = D["jmax"]
    xa, ya = D["xa"], D["ya"]
    sc = K.physical_scales()

    # ---- helicity and helicity-conserving (Taylor) relaxation estimate --------------
    # RMHD relative helicity  K = 2 B0 int (psi_inf - psi) dV  (psi_inf = field-free plateau);
    # for an axisymmetric tube this equals 2 pi L B0 int r^2 B_theta dr.  A linear force-free
    # (uniform-alpha) state of radius R_l carrying the same K has  W = pi L kappa^2/(16 R_l^4),
    # kappa = 4K/(pi L B0)  [Taylor 1974; Bareford, Hood & Browning 2013 for the Bessel version].
    dVc = (W / N) ** 2 * (L / Nc)
    corners = lambda ps: 0.25 * (ps[:, 0, 0] + ps[:, 0, -1] + ps[:, -1, 0] + ps[:, -1, -1])
    Ktime = np.array([2.0 * np.sum((corners(p)[:, None, None] - p)) * dVc for p in psi_all])

    def r_rms(psi):
        Bx_, By_ = K.field_components(psi, W)
        w_ = Bx_**2 + By_**2
        xg = (np.arange(N) - N // 2) * (W / N)
        X_, Y_ = np.meshgrid(xg, xg, indexing="ij")
        out = []
        for k in range(Nc):
            wk = w_[k]
            xc, yc = (wk * X_).sum() / wk.sum(), (wk * Y_).sum() / wk.sum()
            out.append(np.sqrt((wk * ((X_ - xc)**2 + (Y_ - yc)**2)).sum() / wk.sum()))
        return float(np.mean(out))
    rr0, rr1 = r_rms(psi_all[0]), r_rms(psi_all[-1])
    Wf = Em[-1]
    kap_f = 4.0 * Ktime[-1] / (np.pi * L)
    R_theory = (np.pi * L * kap_f**2 / (16.0 * Wf)) ** 0.25          # radius implied by (K_f, W_f)
    R_sim = rr1 * np.sqrt(1.5)       # uniform-alpha state: r_rms^2 = (2/3) R_l^2
    kap0 = 4.0 * Ktime[0] / (np.pi * L)
    W_relax = lambda Rl: np.pi * L * kap0**2 / (16.0 * Rl**4)
    summary["helicity"] = dict(K0=float(Ktime[0]), K_final_over_K0=float(Ktime[-1] / Ktime[0]),
                               K_max_dev=float(np.max(np.abs(Ktime / Ktime[0] - 1))))
    summary["relaxation"] = dict(
        W0=float(Em[0]), W_final=float(Wf), dW_over_W0=float((Em[0] - Wf) / Em[0]),
        R_l_implied_by_Wf_Kf=float(R_theory), R_l_measured_uniform_equiv=float(R_sim),
        dW_over_W0_predicted_at_measured_R=float((Em[0] - W_relax(R_sim)) / Em[0]),
        dW_over_W0_predicted_R=[[float(r), float((Em[0] - W_relax(r)) / Em[0])] for r in (1.0, 1.2, 1.5, 1.8, 2.5)])

    k_pk = int(np.argmax(Ek))                          # peak kinetic energy
    t_pk = t[k_pk]
    isnap0 = 0
    isnap1 = int(np.argmin(np.abs(t_snap - t_pk)))
    isnap2 = len(t_snap) - 1
    isnapv = int(np.argmin(np.abs(t_snap - 72.0)))      # onset of nonlinearity: coherent helical sheet

    # ---- (a,b) 3D field lines before / after
    rings = [(0.15, 4), (0.35, 6), (0.55, 8), (0.75, 10), (0.9, 12)]
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    draw_lines(ax1, psi_all[isnap0], W, L, 1.0, rings,
               "(a) t = %.0f τ_A: twisted equilibrium\n(zero net current, line-tied at z=0,L)" % t_snap[isnap0])
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    draw_lines(ax2, psi_all[isnap1], W, L, 1.0, rings,
               "(b) t = %.0f τ_A: kink  (max E_k)\nfield lines reconnect, axis displaced" % t_snap[isnap1])

    # ---- (c) midplane j_z at three times
    axc = fig.add_subplot(gs[0, 2])
    kmid = Nc // 2
    ext = [-W / 2, W / 2, -W / 2, W / 2]
    jm = np.percentile(np.abs(j_all[isnapv][kmid]), 99.7)
    im = axc.imshow(j_all[isnapv][kmid].T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-jm, vmax=jm)
    axc.contour(np.linspace(-W / 2, W / 2, N), np.linspace(-W / 2, W / 2, N),
                psi_all[isnapv][kmid].T, levels=24, colors="k", linewidths=0.4, alpha=0.6)
    axc.set_title("(c) j$_z$ and field lines at mid-length z=L/2, t=%.0f τ_A" % t_snap[isnapv], fontsize=10)
    axc.set_xlabel("x"); axc.set_ylabel("y")
    plt.colorbar(im, ax=axc, fraction=0.046, pad=0.02)

    # ---- (d) axial cut x-z of j_z
    axd = fig.add_subplot(gs[0, 3])
    jy0 = j_all[isnapv][:, :, N // 2]                  # (Nc, N): plane y ~ 0
    zc = (np.arange(Nc) + 0.5) * L / Nc
    jmd = np.percentile(np.abs(jy0), 99.7)
    imd = axd.imshow(jy0, origin="lower", extent=[-W / 2, W / 2, 0, L], cmap="RdBu_r",
                     vmin=-jmd, vmax=jmd, aspect="auto")
    axd.set_title("(d) axial cut y=0: helical current sheet along the loop", fontsize=10)
    axd.set_xlabel("x"); axd.set_ylabel("z (footpoints at 0 and L)")
    plt.colorbar(imd, ax=axd, fraction=0.046, pad=0.02)

    # ---- (e) energy history
    axe = fig.add_subplot(gs[1, 0])
    axe.plot(t, Em[0] - Em, color="#d62728", lw=2, label=r"magnetic energy released $E_B(0)-E_B$")
    axe.plot(t, Ek, color="#1f77b4", lw=2, label="kinetic energy $E_K$")
    axe.plot(t, Qo, color="#ff7f0e", lw=2, label="cumulative anomalous-Ohmic heat")
    axe.plot(t, Qv, color="#9467bd", lw=2, ls="--", label="cumulative viscous heat")
    tot = Em + Ek + Qo + Qv
    axe.plot(t, (tot - tot[0]), "k:", lw=1.3, label="total drift  ΔE_B+ΔE_K+Q")
    axe.set_xlabel(r"t [$\tau_A$]"); axe.set_ylabel("energy (code units, B0²a³/μ0)")
    axe.set_title("(e) energy conversion and budget", fontsize=10.5)
    axe.legend(fontsize=7.5, loc="center right"); axe.grid(alpha=.3)
    dE = Em[0] - Em[-1]
    axe.text(0.03, 0.97, "released %.0f%% of E$_B$(0)\nOhmic : viscous = %.2f : %.2f\naccounted for: %.0f%% of ΔE$_B$" %
             (100 * dE / Em[0], Qo[-1] / max(Qo[-1] + Qv[-1], 1e-30), Qv[-1] / max(Qo[-1] + Qv[-1], 1e-30),
              100 * (Qo[-1] + Qv[-1] + Ek[-1]) / max(dE, 1e-30)),
             transform=axe.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#888"))
    summary["dEmag_released"] = float(dE)
    summary["dEmag_over_Emag0"] = float(dE / Em[0])
    summary["Q_ohm_final"] = float(Qo[-1]); summary["Q_visc_final"] = float(Qv[-1])
    summary["energy_budget_residual_over_dE"] = float((tot[-1] - tot[0]) / max(dE, 1e-30))
    summary["Ek_max"] = float(Ek.max()); summary["t_peak_Ek"] = float(t_pk)
    summary["jmax_over_jeq"] = float(jmax.max() / meta["jeq"])

    # ---- exponential growth
    axg = fig.add_subplot(gs[1, 1])
    axg.semilogy(t, Ek, color="#1f77b4", lw=2, label="E$_K$(t)")
    g, win = growth_rate_from_run(dict(t=t, Ek=Ek, jmax=jmax, jeq=meta["jeq"]))
    if np.isfinite(g):
        tt = np.linspace(win[0], win[1], 20)
        iref = np.argmin(np.abs(t - win[0]))
        axg.semilogy(tt, Ek[iref] * np.exp(2 * g * (tt - win[0])), "r--", lw=2,
                     label=r"$e^{2\gamma t}$, γ = %.3f τ$_A^{-1}$" % g)
        summary["gamma_production"] = g
    axg.set_xlabel(r"t [$\tau_A$]"); axg.set_ylabel("E$_K$")
    axk = axg.twinx()
    axk.plot(t_snap, Ktime / Ktime[0], color="#2ca02c", lw=2, label="helicity K/K$_0$")
    axk.set_ylim(0.0, 1.15); axk.set_ylabel("K / K$_0$", color="#2ca02c")
    axk.tick_params(axis="y", colors="#2ca02c")
    axg.set_title("(h) exponential kink growth; helicity conservation", fontsize=10)
    h1, l1 = axg.get_legend_handles_labels(); h2, l2 = axk.get_legend_handles_labels()
    axg.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right"); axg.grid(alpha=.3, which="both")

    # ================================================================== coupling to the 1D loop
    # choose three footpoints: axis, mid-radius, and the most strongly heated line
    cand = [(r * np.cos(th), r * np.sin(th)) for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            for th in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    cand = np.array(cand)
    # trace each candidate in every snapshot and integrate heating along the line
    Hline = np.zeros((len(t_snap), len(cand), Nc))
    for ti in range(len(t_snap)):
        Bx, By = K.field_components(psi_all[ti], W)
        path = K.trace_lines(Bx, By, W, L, 1.0, cand[:, 0], cand[:, 1])
        Hline[ti] = K.sample_along(path, H_all[ti], W).T
    heat_int = Hline.sum(axis=(0, 2)) * (t_snap[1] - t_snap[0]) * (L / Nc)   # line-integrated heat
    k_hot = int(np.argmax(heat_int))
    k_axis = 0
    k_mid = int(np.argmin(np.hypot(cand[:, 0] - 0.6, cand[:, 1] - 0.0)))
    lines = [("axis", k_axis), ("r=0.6 line", k_mid), ("hottest line", k_hot)]
    summary["lines"] = {nm: dict(x0=float(cand[k, 0]), y0=float(cand[k, 1]),
                                 heat_int_code=float(heat_int[k])) for nm, k in lines}

    # heating Q(z,t) of the hottest line
    axq = fig.add_subplot(gs[2, 0])
    Qimg = Hline[:, k_hot, :] * sc["Q_unit"]
    imq = axq.imshow(np.log10(np.maximum(Qimg.T, 1e-9)), origin="lower", aspect="auto",
                     extent=[t_snap[0] * sc["tauA"], t_snap[-1] * sc["tauA"], 0, L * sc["a"] / 1e6],
                     cmap="magma", vmin=-6, vmax=np.log10(max(Qimg.max(), 1e-5)))
    axq.set_xlabel("t [s]"); axq.set_ylabel("distance along loop [Mm]")
    axq.set_title("(i) heating Q(s,t) sampled on the hottest field line  [log10 W m$^{-3}$]", fontsize=10)
    plt.colorbar(imq, ax=axq, fraction=0.046, pad=0.02)

    # 1D loop response for the three lines
    Lc_corona = L * sc["a"]                             # 90 Mm, as in Reid et al. 2021
    chromo = 5.0 / 90.0                                 # ~5 Mm chromosphere at each footpoint
    results = {}
    for nm, k in lines:
        loop = P.Loop1D(L_loop=Lc_corona, n_cells=400, chromo_frac=chromo,
                        T_cor=0.8e6, n_cor=1.0e15)
        # relax to a ~1 MK corona with weak uniform heating, then switch it off (Reid et al. 2021)
        Q0 = np.where(loop.corona, 3.0e-5, 0.0)
        while loop.t < 1500.0:
            loop.step(min(loop.cfl_dt(0.3), 2.0), Q0)
        t_start = loop.t
        drv = K.KinkHeatingDriver(loop, t_snap, Hline[:, k, :], sc, L)
        drv.t_phys = drv.t_phys + t_start
        recs = {"t": [], "T": [], "n": [], "v": []}
        next_rec = loop.t
        T_end = t_start + drv.t_phys[-1] - t_start + 1500.0
        while loop.t < T_end:
            dt = min(loop.cfl_dt(0.3), 2.0)
            loop.step(dt, drv.Q_of(loop.t))
            if loop.t >= next_rec:
                rho, v, p, n, T = loop.primitives()
                recs["t"].append(loop.t - t_start); recs["T"].append(T.copy())
                recs["n"].append(n.copy()); recs["v"].append(v.copy())
                next_rec += 10.0
            if not np.isfinite(loop.E).all():
                break
        results[nm] = {kk: np.array(vv) for kk, vv in recs.items()}
        results[nm]["loop"] = loop
    # temperature map for the hottest line
    axt = fig.add_subplot(gs[2, 1])
    rr = results["hottest line"]; lp = rr["loop"]
    imt = axt.imshow(np.log10(rr["T"]), origin="lower", aspect="auto",
                     extent=[0, lp.L / 1e6, rr["t"][0], rr["t"][-1]], cmap="inferno", vmin=5.0, vmax=None)
    axt.set_xlabel("s along loop [Mm]"); axt.set_ylabel("t after relaxation [s]")
    axt.set_title("(j) 1D loop response: log$_{10}$ T(s,t), hottest line", fontsize=10)
    plt.colorbar(imt, ax=axt, fraction=0.046, pad=0.02)

    axa = fig.add_subplot(gs[2, 2])
    colors = {"axis": "#1f77b4", "r=0.6 line": "#2ca02c", "hottest line": "#d62728"}
    for nm, rr in results.items():
        apex = rr["T"].shape[1] // 2
        axa.plot(rr["t"], rr["T"][:, apex] / 1e6, color=colors[nm], lw=2, label=nm)
        summary.setdefault("loop_response", {})[nm] = dict(
            Tapex_max_MK=float(rr["T"][:, apex].max() / 1e6),
            Tapex_initial_MK=float(rr["T"][0, apex] / 1e6),
            n_apex_max=float(rr["n"][:, apex].max()),
            vmax_kms=float(np.abs(rr["v"]).max() / 1e3))
    axa.set_xlabel("t after relaxation [s]"); axa.set_ylabel(r"apex temperature [MK]")
    axa.set_title("(k) loop-apex temperature on three field lines", fontsize=10.5)
    axa.legend(fontsize=8); axa.grid(alpha=.3)

    axp = fig.add_subplot(gs[2, 3])
    Hvol = (np.array([H_all[i].sum() for i in range(len(t_snap))]) * (W / N)**2 * (L / Nc))
    axp.plot(t_snap, Hvol, color="#ff7f0e", lw=2)
    axp.set_xlabel(r"t [$\tau_A$]"); axp.set_ylabel("volume-integrated heating rate (code)")
    E_J = dE * sc["E_unit"]
    axp.set_title("(l) heating rate: ΔE_B = %.2g J (%.2g erg)\nB0=10 G, n=1e15 m$^{-3}$, a=4.5 Mm, L=90 Mm" %
                  (E_J, E_J * 1e7), fontsize=10)
    axp.grid(alpha=.3)
    summary["released_energy_J"] = float(E_J)
    summary["scales"] = {k: float(v) for k, v in sc.items()}

    fig.savefig(out_png, dpi=100, facecolor="white")
    json.dump(summary, open(os.path.join(RES, "kink_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()

