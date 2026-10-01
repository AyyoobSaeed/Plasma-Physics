#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis + single summary figure for scaling_study_2d.py  (reads scaling_results/)."""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "scaling_results")
import tearing_theory as T


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def powerfit(x, y):
    c = np.polyfit(np.log(x), np.log(y), 1)
    return float(c[0]), float(np.exp(c[1]))


def eigen_gamma_max_a(S, a, kgrid=(1, 2, 3, 4, 5, 6, 8)):
    eta = 1.0 / S
    g = [T.growth_rate(k, eta, eta, a, Ny=384) for k in kgrid]
    i = int(np.argmax(g))
    return g[i], kgrid[i]


def main():
    summary = {}
    th = load("theory_eigen.json")
    lin = load("linear_code.json")
    nl = load("nonlinear_scan.json")
    cv = load("convergence.json")

    fig = plt.figure(figsize=(17, 15.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.30, left=0.06, right=0.985,
                          top=0.925, bottom=0.05)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    fig.suptitle("Parametric scaling study of the 2D resistive-MHD reconnection code: "
                 "measured trends vs theory", fontsize=15, fontweight="bold")

    # ------------------------------------------------------------------ (a)
    a_ = ax[0][0]
    if th:
        Sa = np.array([r["Sa"] for r in th])
        gm = np.array([r["gmax"] * r["a"] for r in th])
        km = np.array([r["kamax"] for r in th])
        p_all, c_all = powerfit(Sa, gm)
        hi = Sa >= 300
        p_hi, c_hi = powerfit(Sa[hi], gm[hi])
        loc = [np.log(gm[i + 1] / gm[i]) / np.log(Sa[i + 1] / Sa[i]) for i in range(len(Sa) - 1)]
        q_all, _ = powerfit(Sa, km)
        q_hi, _ = powerfit(Sa[hi], km[hi])
        a_.loglog(Sa, gm, "o-", color="#1f77b4", lw=2, ms=7, label="eigenvalue (periodic double sheet)")
        xs = np.logspace(np.log10(Sa.min()), np.log10(Sa.max()), 50)
        a_.loglog(xs, c_hi * xs**-0.5 * (gm[-1] / (c_hi * Sa[-1]**-0.5)), "--", color="#d62728",
                  lw=1.5, label=r"theory $S_a^{-1/2}$ (anchored at high $S_a$)")
        a_.loglog(xs, c_hi * xs**p_hi, ":", color="#1f77b4", lw=1.3,
                  label=r"fit $S_a\geq300$: slope %.2f" % p_hi)
        a_.set_xlabel(r"$S_a = a v_A/\eta$"); a_.set_ylabel(r"$\gamma_{max}\,a/v_A$")
        a_.set_title("(a) fastest tearing growth rate", fontsize=11)
        a_.legend(fontsize=8); a_.grid(alpha=.3, which="both")
        summary["gamma_max_exponent_all"] = p_all
        summary["gamma_max_exponent_Sa>=300"] = p_hi
        summary["gamma_max_local_slopes"] = [float(v) for v in loc]
        summary["kmax_exponent_all"] = q_all
        summary["kmax_exponent_Sa>=300"] = q_hi
    # ------------------------------------------------------------------ (b)
    b_ = ax[0][1]
    if th:
        for r in th:
            b_.semilogx(np.array(r["ka"]), np.array(r["gamma"]) * r["a"], lw=1.4,
                        label=r"$S_a$=%g" % r["Sa"])
        b_.set_ylim(0, None)
        b_.set_xlabel(r"$ka$"); b_.set_ylabel(r"$\gamma a/v_A$")
        b_.set_title("(b) growth-rate spectrum  $\\gamma(k)$ (eigenvalues)", fontsize=11)
        b_.legend(fontsize=7.5, ncol=2); b_.grid(alpha=.3)
        c_ = ax[0][2]
        c_.loglog(Sa, km, "o-", color="#2ca02c", lw=2, ms=7, label=r"eigenvalue $k_{max}a$")
        c_.loglog(xs, km[-1] * (xs / Sa[-1])**-0.25, "--", color="#d62728", lw=1.5,
                  label=r"theory $S_a^{-1/4}$")
        c_.set_xlabel(r"$S_a$"); c_.set_ylabel(r"$k_{max}a$")
        c_.set_title("(c) wavenumber of fastest mode (fit: %.2f)" % q_hi, fontsize=11)
        c_.legend(fontsize=8); c_.grid(alpha=.3, which="both")

    # ------------------------------------------------------------------ (d) code vs eigen
    d_ = ax[1][0]
    rows = []
    if lin:
        for r in lin:
            eta, N, a = r["eta"], r["N"], r["a"]
            Ny = 256 if eta >= 4e-4 else 512
            ge = np.array([T.growth_rate(m, eta, eta, a, Ny=Ny) for m in r["modes"]])
            gmx = ge.max()
            for m, gc, fl, g0 in zip(r["modes"], r["gamma"], r["flags"], ge):
                clean = (fl == "ok") and (g0 >= 0.7 * gmx) and np.isfinite(gc)
                rows.append(dict(eta=eta, N=N, m=m, gc=gc, ge=float(g0), clean=bool(clean)))
        cl = [x for x in rows if x["clean"]]
        for N_, col, mk in ((128, "#9467bd", "s"), (192, "#ff7f0e", "^"), (256, "#1f77b4", "o")):
            pts = [x for x in cl if x["N"] == N_]
            if pts:
                d_.plot([p["ge"] for p in pts], [p["gc"] for p in pts], mk, color=col, ms=8,
                        mfc="none", mew=2, label="code, N=%d" % N_)
        allg = [p["ge"] for p in cl] or [0.1]
        lim = (0, 1.1 * max(allg + [p["gc"] for p in cl]))
        d_.plot(lim, lim, "k--", lw=1, label="y = x")
        d_.set_xlim(lim); d_.set_ylim(lim)
        err = np.array([abs(p["gc"] - p["ge"]) / p["ge"] for p in cl])
        d_.set_xlabel(r"$\gamma$ from linear eigenvalue solver")
        d_.set_ylabel(r"$\gamma$ measured in the nonlinear code")
        d_.set_title("(d) code verification: %d clean modes, median err %.1f%%, max %.1f%%" %
                     (len(cl), 100 * np.median(err) if err.size else np.nan,
                      100 * err.max() if err.size else np.nan), fontsize=10)
        d_.legend(fontsize=8); d_.grid(alpha=.3)
        summary["linear_code_vs_eigen"] = dict(
            n=len(cl), median_rel_err=float(np.median(err)) if err.size else None,
            max_rel_err=float(err.max()) if err.size else None,
            rows=[{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                   for k, v in x.items()} for x in rows])

    # ------------------------------------------------------------------ nonlinear
    if nl:
        a_nl = nl[0]["a"]
        Ss, tons, nOs, gmx_s, kmx_s = [], [], [], [], []
        samp_all = []
        Rvals, fohm, Mpk = [], [], []
        for r in nl:
            t = np.array(r["t"]); psi = np.array(r["psirec"])
            S = r["S"]
            thr = 0.05                                   # absolute reconnected-flux threshold (N-independent definition)
            idx = np.where(psi >= thr)[0]
            ton = t[idx[0]] if idx.size else np.nan
            g, kk = eigen_gamma_max_a(S, a_nl)
            Ss.append(S); tons.append(ton); gmx_s.append(g); kmx_s.append(kk)
            nO = np.array(r["nO"]); sel = (t >= ton) & (t <= ton + 4.0 / g)
            nOs.append(float(np.median(nO[sel])) if sel.any() else np.nan)
            for smp in r["samples"]:
                if smp["t"] >= ton and smp["resolved"] and smp["Bup"] > 0.2:
                    samp_all.append((S, smp["delta"], smp["ell"], smp["Bup"], smp["t"] - ton))
            h = r["hist"]
            Eb, Ek, Qo, Qv = (np.array(h[k]) for k in ("E_mag", "E_kin", "Q_ohm", "Q_visc"))
            # work done by the external E-field that holds the Harris sheet:
            #   dW_F/dt = eta <j j0>   (cumulative trapezoid over the frames)
            PF = np.array(r["PF"])
            WF = np.concatenate([[0.0], np.cumsum(0.5 * (PF[1:] + PF[:-1]) * np.diff(t))])
            res = (Eb + Ek + Qo + Qv - WF) / Eb[0] - 1.0
            r["_resid"] = float(np.abs(res).max())
            fohm.append((S, float(Qo[-1] / (Qo[-1] + Qv[-1])), float(Qv[-1] / (Qo[-1] + Qv[-1]))))
            MX = np.array(r["MX"])
            Mpk.append((S, float(np.max(MX[t >= ton])) if np.any(t >= ton) else np.nan,
                        1.0 / (S * a_nl)))
        Ss = np.array(Ss, float); tons = np.array(tons); gmx_s = np.array(gmx_s)
        e_ = ax[1][1]
        ok = np.isfinite(tons)
        p_t, _ = powerfit(Ss[ok], tons[ok])
        p_th = -np.polyfit(np.log(Ss), np.log(gmx_s), 1)[0]           # t ~ 1/gamma_max
        e_.loglog(Ss[ok], tons[ok], "o-", color="#d62728", lw=2, ms=8, label="measured $t_{onset}$ (slope %.2f)" % p_t)
        e_.loglog(Ss, tons[ok][0] * (gmx_s[0] / gmx_s) * (1 if ok[0] else 1), "s--", color="#1f77b4",
                  lw=1.5, label=r"$\propto 1/\gamma_{max}^{eig}$ (slope %.2f)" % p_th)
        e_.loglog(Ss, tons[ok][0] * (Ss / Ss[0])**0.5, ":", color="gray", label=r"asymptotic $S^{1/2}$")
        e_.set_xlabel("S = 1/η"); e_.set_ylabel(r"onset time  [$\tau_A$]")
        e_.set_title("(e) onset time vs S", fontsize=11); e_.legend(fontsize=8); e_.grid(alpha=.3, which="both")
        summary["t_onset"] = dict(S=Ss.tolist(), t=tons.tolist(), slope_measured=p_t,
                                  slope_expected_from_eigen=p_th)

        # (f) local Sweet-Parker
        f_ = ax[1][2]
        if samp_all:
            sa = np.array(samp_all)
            Sell = sa[:, 2] * sa[:, 3] / (1.0 / sa[:, 0])
            ratio = sa[:, 1] / sa[:, 2]
            sc = f_.scatter(Sell, ratio, c=np.log10(sa[:, 0]), cmap="viridis", s=14, alpha=.7)
            p_sp, c_sp = powerfit(Sell, ratio)
            xs2 = np.logspace(np.log10(Sell.min()), np.log10(Sell.max()), 30)
            f_.loglog(xs2, c_sp * xs2**p_sp, "r-", lw=2, label="fit slope %.2f" % p_sp)
            f_.loglog(xs2, ratio.mean() * (xs2 / np.exp(np.log(Sell).mean()))**-0.5, "k--", lw=1.5,
                      label=r"Sweet-Parker $S_\ell^{-1/2}$")
            f_.set_xscale("log"); f_.set_yscale("log")
            cb = fig.colorbar(sc, ax=f_, fraction=0.045, pad=0.02); cb.set_label("log10 S", fontsize=8)
            # CONFOUNDING CHECK: separate the dependences.  Sweet-Parker needs
            #   delta = ell^{1/2} B_up^{-1/2} S^{-1/2}   (exponents +0.5, -0.5, -0.5).
            X = np.column_stack([np.ones(len(sa)), np.log(sa[:, 2]), np.log(sa[:, 0]), np.log(sa[:, 3])])
            cf = np.linalg.lstsq(X, np.log(sa[:, 1]), rcond=None)[0]
            d_sp = sa[:, 2] / np.sqrt(Sell)
            f_.text(0.03, 0.04,
                    "single-variable slope %.2f looks like SP, but the\nregression  δ ∝ ℓ^%.2f S^%.2f B^%.2f\n(SP: ℓ^+0.5 S^-0.5 B^-0.5) and δ/δ_SP = %.1f\nshow it is a coincidence" %
                    (p_sp, cf[1], cf[2], cf[3], float(np.median(sa[:, 1] / d_sp))),
                    transform=f_.transAxes, fontsize=7.5, va="bottom",
                    bbox=dict(boxstyle="round", fc="#fff3cd", ec="#c9a227"))
            f_.set_xlabel(r"$S_\ell=\ell B_{up}/\eta$"); f_.set_ylabel(r"$\delta/\ell$")
            f_.set_title("(f) local sheet aspect ratio (n=%d): NOT Sweet-Parker" % len(sa), fontsize=10.5)
            f_.legend(fontsize=8, loc="upper right"); f_.grid(alpha=.3, which="both")
            summary["local_SP"] = dict(slope_single_variable=p_sp, n=len(sa), expected=-0.5,
                                       regression_exponents=dict(ell=float(cf[1]), S=float(cf[2]),
                                                                 Bup=float(cf[3])),
                                       sp_exponents=dict(ell=0.5, S=-0.5, Bup=-0.5),
                                       median_delta_over_delta_SP=float(np.median(sa[:, 1] / d_sp)),
                                       verdict="single-variable slope coincides with -1/2 but the multi-variable "
                                               "regression contradicts Sweet-Parker; not demonstrated")

        # (g) island number
        g_ = ax[2][0]
        nOs = np.array(nOs)
        g_.semilogx(Ss, nOs, "o-", color="#2ca02c", lw=2, ms=8, label="median $N_O$ after onset")
        g_.semilogx(Ss, 2 * np.array(kmx_s) * 1.0, "s--", color="#1f77b4", lw=1.5,
                    label=r"linear theory $2\,k_{max}L/2\pi$ (2 sheets)")
        g_.set_xlabel("S = 1/η"); g_.set_ylabel("number of O-points (islands)")
        g_.set_title("(g) self-selected island number vs S", fontsize=11)
        g_.legend(fontsize=8); g_.grid(alpha=.3, which="both"); g_.set_ylim(0, None)
        summary["islands"] = dict(S=Ss.tolist(), nO_median=nOs.tolist(), nO_linear=[2 * k for k in kmx_s])

        # (i) X-point reconnection rate vs S and energy-budget closure
        i_ = ax[2][2]
        Mp = np.array(Mpk)
        okM = np.isfinite(Mp[:, 1])
        if okM.any():
            p_M, _ = powerfit(Mp[okM, 0], Mp[okM, 1])
            i_.loglog(Mp[okM, 0], Mp[okM, 1], "o-", color="#ff7f0e", lw=2, ms=8,
                      label=r"peak $M=\eta j_X/B_{up}^2$ (slope %.2f)" % p_M)
            i_.loglog(Mp[:, 0], Mp[:, 2], "s:", color="gray", lw=1.5,
                      label=r"held-sheet baseline $1/S_a$")
            i_.loglog(Mp[:, 0], Mp[okM, 1][0] * (Mp[:, 0] / Mp[okM, 0][0])**-0.5, "k--", lw=1.2,
                      label=r"$S^{-1/2}$")
            i_.set_xlabel("S = 1/η"); i_.set_ylabel("normalised reconnection rate M")
            resid = max(r["_resid"] for r in nl)
            i_.set_title("(i) X-point rate vs S  (energy closure: max |resid| = %.1e)" % resid, fontsize=10)
            i_.legend(fontsize=8); i_.grid(alpha=.3, which="both")
            summary["M_peak"] = dict(S=Mp[:, 0].tolist(), M=Mp[:, 1].tolist(), slope=p_M)
        summary["energy_closure_max_residual"] = {int(r["S"]): r["_resid"] for r in nl}
        summary["heat_partition_Qohm_Qvisc_fractions"] = {int(s): [o, v] for s, o, v in fohm}

    # ------------------------------------------------------------------ (h) convergence
    h_ = ax[2][1]
    if cv:
        for r, col in zip(cv, ("#9467bd", "#ff7f0e", "#1f77b4")):
            h_.plot(r["t"], r["psirec"], color=col, lw=2, label="N=%d" % r["N"])
        h_.set_xlabel(r"t [$\tau_A$]"); h_.set_ylabel(r"reconnected flux $\Psi_{rec}$")
        h_.set_title("(h) resolution convergence at S=%d" % cv[0]["S"], fontsize=11)
        h_.legend(fontsize=8); h_.grid(alpha=.3)
        fin = {}
        for r in cv:
            t_ = np.array(r["t"]); ps_ = np.array(r["psirec"]); hh = r["hist"]
            PF_ = np.array(r["PF"])
            WF_ = np.concatenate([[0.0], np.cumsum(0.5 * (PF_[1:] + PF_[:-1]) * np.diff(t_))])
            res_ = (np.array(hh["E_mag"]) + np.array(hh["E_kin"]) + np.array(hh["Q_ohm"]) +
                    np.array(hh["Q_visc"]) - WF_) / hh["E_mag"][0] - 1.0
            tc = lambda x: float(t_[np.where(ps_ >= x)[0][0]]) if np.any(ps_ >= x) else float("nan")
            fin[r["N"]] = dict(t_psi_0p05=tc(0.05), t_psi_0p2=tc(0.2), t_psi_0p4=tc(0.4),
                               psirec_end=float(ps_[-1]), jmax_max=float(max(r["jmax"])),
                               closure_resid=float(np.abs(res_).max()))
        summary["convergence"] = fin
        h_.text(0.03, 0.97, "onset t(Ψ=0.2): " + ", ".join("N=%d: %.1f" % (n, v["t_psi_0p2"]) for n, v in fin.items()) +
                "\nearly phase converged for N≥192;\nsaturation level/speed still N-dependent",
                transform=h_.transAxes, va="top", fontsize=7.5, bbox=dict(boxstyle="round", fc="white", ec="#888"))

    fig.savefig(os.path.join(HERE, "scaling_study_2d.png"), dpi=110, facecolor="white")
    json.dump(summary, open(os.path.join(RES, "scaling_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "linear_code_vs_eigen"}, indent=1)[:6000])
    if "linear_code_vs_eigen" in summary:
        s = summary["linear_code_vs_eigen"]
        print("linear code vs eigen: n=%s median err=%s max err=%s" % (s["n"], s["median_rel_err"], s["max_rel_err"]))


if __name__ == "__main__":
    main()
