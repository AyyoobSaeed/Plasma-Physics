#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Heating analysis: which forms of heating contribute, and how do they evolve?
============================================================================
Reads the stored results (scaling_results/) -- no simulation is re-run.

Forms of heating that exist in these models (and ones that do not)
------------------------------------------------------------------
Both MHD codes are INCOMPRESSIBLE reduced MHD, so the only irreversible heating channels are
    Ohmic      H_ohm  = eta(j) j^2        (3D kink: anomalous eta_c above j_crit, eta_b = 0;
                                           2D: uniform eta = 1/S)
    viscous    H_visc = nu w^2             (w = parallel vorticity)
Not present: compressive / shock heating (p dV, shock viscosity -- the dominant channel in Lare3d
when eta_b = 0), Hall / kinetic heating, and adiabatic heating.  Conduction and radiation are NOT
heating: they only appear in the 1D loop, which receives H as an input.

3D kink (kink_production.npz)
  (a) rate of each form vs time (from snapshots, checked against d/dt of the running integrals)
  (b) cumulative heat of each form and the cumulative Ohmic fraction
  (c) rate budget  -dE_B/dt = P_ohm + P_visc + dE_K/dt   (which channel the released energy takes)
  (d) where the heat goes radially, per form (time-integrated)
  (e) where the heat goes along the loop, per form (line-tying: footpoints vs apex)
  (f) intermittency: share of the heating in the top 1 % of cells, and volume with |j| > j_crit

2D held-sheet scan (nonlinear_scan.json)
  (g) cumulative heat at S=2000, split by ENERGY SOURCE: external forcing that holds the sheet
      (W_F) | released field energy -> Ohmic (Q_ohm - W_F) | released field energy -> viscous (Q_visc)
  (h) end-of-run fractions vs S
  (i) instantaneous viscous share of the heating vs time, all S

Usage:  python heating_analysis.py     ->  heating_analysis.png, scaling_results/heating_summary.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "scaling_results")
C_OHM, C_VIS, C_TOT, C_EB, C_EK = "#ff7f0e", "#9467bd", "k", "#d62728", "#1f77b4"


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ma(x, n=5):
    """centred moving average (n odd), edge-padded"""
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, (n // 2, n // 2), mode="edge"), np.ones(n) / n, mode="valid")


def trap_weights(t):
    w = np.zeros_like(t)
    d = np.diff(t)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


# =============================================================================== 3D kink
def kink_analysis(axes, summary):
    path = os.environ.get("KINK_PROD", os.path.join(RES, "kink_production.npz"))
    if not os.path.exists(path):
        for a in axes:
            a.text(0.5, 0.5, "kink_production.npz not found", ha="center", transform=a.transAxes)
        return
    D = np.load(path, allow_pickle=True)
    meta = json.loads(str(D["meta"]))
    N, Nc, W, L = meta["N"], meta["Nc"], meta["W"], meta["L"]
    jc, eta_c = meta["j_crit"], meta["eta_c"]
    dj = 0.08 * jc                       # as set in kink_production.run_prod (not stored in meta)
    t, Em, Ek, Qo, Qv, jmax = (D[k] for k in ("t", "Em", "Ek", "Qo", "Qv", "jmax"))
    ts, j_all, H_all = D["t_snap"], D["j"], D["H"]
    dx, dz = W / N, L / Nc
    dV = dx * dx * dz
    x1 = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x1, x1, indexing="ij")
    rr = np.hypot(X, Y)
    wts = trap_weights(ts)

    ns = len(ts)
    Po, Pv = np.zeros(ns), np.zeros(ns)
    top1, vol_act = np.zeros(ns), np.zeros(ns)
    acc_o = np.zeros((Nc, N, N)); acc_v = np.zeros((Nc, N, N))
    for i in range(ns):
        j = j_all[i].astype(float); H = H_all[i].astype(float)
        Ho = eta_c * 0.5 * (1.0 + np.tanh((np.abs(j) - jc) / dj)) * j**2
        Hv = np.clip(H - Ho, 0.0, None)
        Po[i], Pv[i] = Ho.sum() * dV, Hv.sum() * dV
        acc_o += wts[i] * Ho; acc_v += wts[i] * Hv
        flat = H.ravel(); k = int(0.99 * flat.size)
        top1[i] = np.partition(flat, k)[k:].sum() / max(flat.sum(), 1e-300)
        vol_act[i] = float(np.mean(np.abs(j) > jc))
    Ptot = Po + Pv
    Eo_snap, Ev_snap = float(wts @ Po), float(wts @ Pv)

    # ---- characteristic times
    i_nl = np.where(jmax / meta["jeq"] >= 2.0)[0]
    t_nl = float(t[i_nl[0]]) if i_nl.size else np.nan
    Qt = Qo + Qv
    t10, t50, t90 = (float(np.interp(f * Qt[-1], Qt, t)) for f in (0.1, 0.5, 0.9))
    t_pk = float(ts[np.argmax(Ptot)])

    # ---- (a) rates
    ax = axes[0]
    ax.plot(t, ma(np.gradient(Qo, t)), color=C_OHM, lw=1.6, label="Ohmic  $d Q_{ohm}/dt$")
    ax.plot(t, ma(np.gradient(Qv, t)), color=C_VIS, lw=1.6, label="viscous  $d Q_{visc}/dt$")
    ax.plot(ts, Po, "o", color=C_OHM, ms=3.5, mfc="none", label="snapshot ∫η(j)j²dV")
    ax.plot(ts, Pv, "s", color=C_VIS, ms=3.5, mfc="none", label="snapshot ∫νw²dV")
    ax.plot(ts, Ptot, "-", color=C_TOT, lw=1.0, alpha=.7, label="total (snapshots)")
    ax.axvline(t_nl, color="gray", ls="--", lw=1); ax.text(t_nl, ax.get_ylim()[1] * .02, " |j|=2j$_{eq}$", fontsize=7, color="gray")
    ax.set_xlabel(r"t [$\tau_A$]"); ax.set_ylabel("heating rate (code units)")
    ax.set_title("(a) 3D kink: heating rate of each form", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # ---- (b) cumulative + Ohmic fraction
    ax = axes[1]
    ax.plot(t, Qo, color=C_OHM, lw=2, label="cumulative Ohmic")
    ax.plot(t, Qv, color=C_VIS, lw=2, ls="--", label="cumulative viscous")
    ax.axvspan(t10, t90, color="gold", alpha=.18, label=r"10-90%% of the heat: %.0f $\tau_A$" % (t90 - t10))
    ax.set_xlabel(r"t [$\tau_A$]"); ax.set_ylabel("cumulative heat (code units)")
    a2 = ax.twinx()
    ok = Qt > 0.01 * Qt[-1]
    a2.plot(t[ok], (Qo / Qt)[ok], color="#2ca02c", lw=1.6, label="Ohmic share of cumulative heat")
    a2.set_ylim(0, 1); a2.set_ylabel("Ohmic fraction", color="#2ca02c"); a2.tick_params(axis="y", colors="#2ca02c")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("(b) cumulative heat by form", fontsize=10); ax.grid(alpha=.3)

    # ---- (c) rate budget
    ax = axes[2]
    dEm = -np.gradient(Em, t); dEk = np.gradient(Ek, t)
    Po_t, Pv_t = np.gradient(Qo, t), np.gradient(Qv, t)
    ax.plot(t, ma(dEm), color=C_EB, lw=2, label=r"released field power $-dE_B/dt$")
    ax.plot(t, ma(Po_t), color=C_OHM, lw=1.5, label="Ohmic")
    ax.plot(t, ma(Pv_t), color=C_VIS, lw=1.5, label="viscous")
    ax.plot(t, ma(dEk), color=C_EK, lw=1.5, label=r"kinetic $dE_K/dt$")
    ax.plot(t, ma(Po_t + Pv_t + dEk), "k:", lw=1.6, label="Ohmic + visc + $dE_K/dt$")
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel(r"t [$\tau_A$]"); ax.set_ylabel("power (code units)")
    ax.set_title("(c) rate budget: where the released energy goes", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    resid = float(np.sqrt(np.mean((dEm - (Po_t + Pv_t + dEk))**2)) / max(np.max(np.abs(dEm)), 1e-12))

    # ---- (d) radial distribution
    ax = axes[3]
    edges = np.linspace(0.0, 2.5, 26); ctr = 0.5 * (edges[1:] + edges[:-1])
    def radial(acc):
        out = np.array([acc[:, (rr >= a) & (rr < b)].sum() * dV for a, b in zip(edges[:-1], edges[1:])])
        return out / np.diff(edges)
    ro, rv = radial(acc_o), radial(acc_v)
    ax.step(ctr, ro, where="mid", color=C_OHM, lw=2, label="Ohmic")
    ax.step(ctr, rv, where="mid", color=C_VIS, lw=2, ls="--", label="viscous")
    ax.axvline(1.0, color="gray", ls=":", lw=1); ax.text(1.02, ax.get_ylim()[1] * .9, "tube edge r=1", fontsize=7, color="gray")
    ax.set_xlabel("r from the initial axis"); ax.set_ylabel("time-integrated heat per unit r")
    ax.set_title("(d) radial deposition of the heat", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # ---- (e) axial distribution
    ax = axes[4]
    zc = (np.arange(Nc) + 0.5) * dz
    zo, zv = acc_o.sum(axis=(1, 2)) * dx * dx / dz, acc_v.sum(axis=(1, 2)) * dx * dx / dz
    ax.plot(zc, zo, color=C_OHM, lw=2, label="Ohmic"); ax.plot(zc, zv, color=C_VIS, lw=2, ls="--", label="viscous")
    ax.set_xlabel("z  (footpoints at 0 and L)"); ax.set_ylabel("time-integrated heat per unit length")
    ax.set_title("(e) axial deposition of the heat", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # ---- (f) intermittency
    ax = axes[5]
    ax.plot(ts, 100 * top1, color="#8c564b", lw=2, label="share of heating in hottest 1 % of cells [%]")
    ax.set_xlabel(r"t [$\tau_A$]"); ax.set_ylabel("%", color="#8c564b"); ax.set_ylim(0, 100)
    a2 = ax.twinx()
    a2.plot(ts, 100 * vol_act, color="#17becf", lw=2, label="volume with |j| > j$_{crit}$ [%]")
    a2.set_ylabel("% of volume", color="#17becf"); a2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax.set_title("(f) how localised is the heating?", fontsize=10); ax.grid(alpha=.3)

    r_core = lambda prof: float(prof[ctr < 1.0].sum() / max(prof.sum(), 1e-300))
    foot = (zc < 0.1 * L) | (zc > 0.9 * L)
    summary["kink"] = dict(
        Q_ohm_final=float(Qo[-1]), Q_visc_final=float(Qv[-1]),
        ohmic_fraction_final=float(Qo[-1] / Qt[-1]),
        ohmic_fraction_at_t10_t50_t90=[float(np.interp(tt, t, Qo / np.maximum(Qt, 1e-300))) for tt in (t10, t50, t90)],
        t_onset_2jeq=t_nl, t_heat_10_50_90=[t10, t50, t90], t_peak_total_rate=t_pk,
        t_peak_ohmic_rate=float(ts[np.argmax(Po)]), t_peak_visc_rate=float(ts[np.argmax(Pv)]),
        snapshot_integral_vs_running_Qo=float(Eo_snap / Qo[-1]), snapshot_integral_vs_running_Qv=float(Ev_snap / Qv[-1]),
        rate_budget_rms_residual_over_peak=resid,
        core_r_lt_1_fraction=dict(ohmic=r_core(ro), viscous=r_core(rv)),
        footpoint_10pct_each_end_fraction=dict(ohmic=float(zo[foot].sum() / zo.sum()), viscous=float(zv[foot].sum() / zv.sum())),
        top1pct_cell_share_at_peak=float(top1[np.argmax(Ptot)]), top1pct_cell_share_median=float(np.median(top1[ts > t_nl])),
        active_volume_pct_max=float(100 * vol_act.max()),
        peak_ohmic_power_share_of_total_rate=float(Po[np.argmax(Ptot)] / Ptot.max()),
        note="no compressive/shock heating exists in this RMHD model; Ohmic is anomalous (eta_c above j_crit), eta_b = 0")


# =============================================================================== 2D scan
def j0_meansq(N, a, L=2 * np.pi):
    """<j0^2> of the held double-Harris sheet (same construction as LinearTearingMHD)."""
    y = np.arange(N) * L / N
    Bx = np.tanh((y - 0.25 * L) / a) - np.tanh((y - 0.75 * L) / a)
    Bx -= Bx.mean()
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    dB = np.fft.ifft(1j * ky * np.fft.fft(Bx)).real
    return float(np.mean(dB**2))


def scan2d_analysis(axes, summary):
    nl = load("nonlinear_scan.json")
    if not nl:
        for a in axes:
            a.text(0.5, 0.5, "nonlinear_scan.json not found", ha="center", transform=a.transAxes)
        return
    rows, out = [], {}
    for r in nl:
        h = r["hist"]; t = np.array(r["t"]) if len(r["t"]) == len(h["t"]) else np.array(h["t"])
        Qo, Qv = np.array(h["Q_ohm"]), np.array(h["Q_visc"])
        PF = np.array(r["PF"])
        WF = np.concatenate([[0.0], np.cumsum(0.5 * (PF[1:] + PF[:-1]) * np.diff(t))])
        base = r["eta"] * j0_meansq(r["N"], r["a"]) * t          # eta<j0^2> t : dissipation of the held sheet itself
        rows.append(dict(S=r["S"], t=t, Qo=Qo, Qv=Qv, WF=WF, base=base, Eb=np.array(h["E_mag"]), Ek=np.array(h["E_kin"])))
        tot = Qo[-1] + Qv[-1]
        field_heat = (Qo[-1] - WF[-1]) + Qv[-1]
        dEB = rows[-1]["Eb"][0] - rows[-1]["Eb"][-1]
        out[int(r["S"])] = dict(
            Q_ohm=float(Qo[-1]), Q_visc=float(Qv[-1]), W_forcing=float(WF[-1]),
            ohmic_fraction=float(Qo[-1] / tot), viscous_fraction=float(Qv[-1] / tot),
            forcing_fed_fraction_of_total_heat=float(WF[-1] / tot),
            field_fed_ohmic=float(Qo[-1] - WF[-1]), field_fed_viscous=float(Qv[-1]),
            viscous_fraction_of_field_fed_heat=float(Qv[-1] / field_heat) if field_heat > 0 else None,
            heat_from_field_over_dEB=float(field_heat / dEB) if dEB > 0 else None,
            held_sheet_baseline_eta_j0sq_t=float(base[-1]), baseline_over_Qohm=float(base[-1] / Qo[-1]))
    summary["scan2d"] = out

    # ---- (g) S=2000 energy-source decomposition
    ax = axes[0]
    R = next((x for x in rows if x["S"] == 2000), rows[len(rows) // 2])
    ax.plot(R["t"], R["Qo"], color=C_OHM, lw=2, label="Ohmic heat $Q_{ohm}$ (total)")
    ax.plot(R["t"], R["WF"], color="#2ca02c", lw=1.6, ls="-.", label="  fed by the external forcing $W_F$")
    ax.plot(R["t"], R["Qo"] - R["WF"], color="#bcbd22", lw=1.6, label="  fed by released field energy  $Q_{ohm}-W_F$")
    ax.plot(R["t"], R["Qv"], color=C_VIS, lw=2, ls="--", label="viscous heat $Q_{visc}$")
    ax.plot(R["t"], R["Eb"][0] - R["Eb"], color=C_EB, lw=1.2, alpha=.6, label="released field energy")
    ax.set_xlabel(r"t [$\tau_A$]"); ax.set_ylabel("cumulative energy (per unit area)")
    ax.set_title("(g) 2D S=%d: heat by energy source" % R["S"], fontsize=10); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)

    # ---- (h) end-of-run fractions vs S
    ax = axes[1]
    S = np.array([x["S"] for x in rows], float)
    tot = np.array([x["Qo"][-1] + x["Qv"][-1] for x in rows])
    ax.semilogx(S, [x["Qo"][-1] for x in rows] / tot, "o-", color=C_OHM, lw=2, label="Ohmic / total heat")
    ax.semilogx(S, [x["Qv"][-1] for x in rows] / tot, "s--", color=C_VIS, lw=2, label="viscous / total heat")
    ax.semilogx(S, [x["WF"][-1] for x in rows] / tot, "^-.", color="#2ca02c", lw=1.6, label="forcing-fed share ($W_F$ / total)")
    ax.set_ylim(0, 1.05); ax.set_xlabel("S = 1/η"); ax.set_ylabel("fraction of total heat")
    ax.set_title("(h) 2D: heating mix at the end of each run", fontsize=10); ax.legend(fontsize=7.5); ax.grid(alpha=.3, which="both")

    # ---- (i) instantaneous viscous share vs time
    ax = axes[2]
    cm = plt.get_cmap("viridis")
    for k, x in enumerate(rows):
        Po, Pv = ma(np.gradient(x["Qo"], x["t"]), 11), ma(np.gradient(x["Qv"], x["t"]), 11)
        # the held sheet's steady dissipation dominates Po until onset, so show the share of the *excess*
        # over the baseline only after the heating departs from it
        sh = Pv / np.maximum(Po + Pv, 1e-30)
        ax.plot(x["t"], sh, color=cm(k / max(len(rows) - 1, 1)), lw=1.6, label="S=%d" % x["S"])
    ax.set_ylim(0, 1); ax.set_xlabel(r"t [$\tau_A$]"); ax.set_ylabel(r"$P_{visc}/(P_{ohm}+P_{visc})$")
    ax.set_title("(i) 2D: instantaneous viscous share of the heating", fontsize=10); ax.legend(fontsize=7.5); ax.grid(alpha=.3)


def main():
    summary = {}
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.36, wspace=0.34, left=0.06, right=0.955, top=0.925, bottom=0.055)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    fig.suptitle("Heating analysis: Ohmic vs viscous contributions and their evolution "
                 "(3D line-tied kink, row 1–2;  2D held-sheet tearing scan, row 3)", fontsize=14, fontweight="bold")
    kink_analysis([ax[0][0], ax[0][1], ax[0][2], ax[1][0], ax[1][1], ax[1][2]], summary)
    scan2d_analysis([ax[2][0], ax[2][1], ax[2][2]], summary)
    fig.savefig(os.path.join(HERE, "heating_analysis.png"), dpi=100, facecolor="white")
    json.dump(summary, open(os.path.join(RES, "heating_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
