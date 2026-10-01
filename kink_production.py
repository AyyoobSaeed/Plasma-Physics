#!/usr/bin/env python3
"""Nonlinear production run + refinement/verification jobs for kink_loop_3d.py

  python kink_production.py prod          # nonlinear kink, anomalous resistivity, snapshots
  python kink_production.py refine        # threshold refinement, L-scaling, resolution check
"""
import os, sys, json, time
os.environ.setdefault("MHD_FFT_WORKERS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "scaling_results")
os.makedirs(OUT, exist_ok=True)
import kink_loop_3d as K


def run_prod(Phi0=7 * np.pi, t_end=170.0, N=80, Nc=40, nu=2.0e-4, tag=""):
    """Nonlinear kink with Lare3d-style anomalous resistivity.  nu=2e-4 matches the
    stability scans (nu=1e-3 stabilises this loop at 6 pi -- see convergence tests)."""
    t0 = time.time()
    s = K.KinkRMHD3D(N=N, W=5.0, L=20.0, Nc=Nc, Phi0=Phi0, nu=nu, eta_b=0.0,
                     eta_c=1.0e-3, j_crit=1.0, eps=0.01, p_prof=3)
    s.j_crit = 1.8 * s.jeq_max
    s.dj = 0.08 * s.j_crit       # sharp switch: tail leakage at the equilibrium current < 1e-4
    snap_times = list(np.arange(0.0, t_end + 1e-9, 2.0))
    r = s.run(t_end, record_dt=1.0, snap_times=snap_times, verbose=True)
    snaps = r.pop("snaps")
    ts = sorted(snaps)
    np.savez_compressed(
        os.path.join(OUT, "kink_production%s.npz" % tag),
        t_snap=np.array([snaps[t]["t"] for t in ts]),
        psi=np.array([snaps[t]["psi"] for t in ts]),
        j=np.array([snaps[t]["j"] for t in ts]),
        H=np.array([snaps[t]["H"] for t in ts]),
        **{k: v for k, v in r.items()},
        meta=json.dumps(dict(Phi0=Phi0, N=N, Nc=Nc, W=5.0, L=20.0, nu=nu, eta_c=1.0e-3,
                             j_crit=s.j_crit, jeq=s.jeq_max, lam=s.lam, p_prof=3,
                             wall=time.time() - t0)))
    print("production run saved (%.0f s)" % (time.time() - t0))


def stage_refine():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [dict(Phi0=4.25 * np.pi, t_end=500.0, p_prof=3),
            dict(Phi0=4.5 * np.pi, t_end=500.0, p_prof=3),
            dict(Phi0=4.75 * np.pi, t_end=450.0, p_prof=3),
            dict(Phi0=6 * np.pi, t_end=300.0, p_prof=3, L=40.0, Nc=64),        # L-scaling
            dict(Phi0=6 * np.pi, t_end=150.0, p_prof=3, N=96, Nc=48),         # resolution
            dict(Phi0=6 * np.pi, t_end=150.0, p_prof=3, nu=1.0e-4)]            # viscosity
    out = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(K.scan_job, **kw) for kw in jobs]
        for kw, f in zip(jobs, futs):
            r = f.result()
            tag = {k: (round(v / np.pi, 2) if k == "Phi0" else v) for k, v in kw.items()}
            print("  done", tag, " t_last=%.0f Ek_last=%.2e" % (r["t"][-1], r["Ek"][-1]), flush=True)
            r["job"] = {k: float(v) for k, v in kw.items()}
            out.append(r)
            json.dump(out, open(os.path.join(OUT, "kink_refine.json"), "w"))


def stage_refine2():
    """Near-threshold bisection with long runs (growth is slow close to Phi_c)."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [dict(Phi0=4.0 * np.pi, t_end=700.0, p_prof=3),
            dict(Phi0=4.1 * np.pi, t_end=700.0, p_prof=3),
            dict(Phi0=4.15 * np.pi, t_end=700.0, p_prof=3)]
    out = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(K.scan_job, **kw) for kw in jobs]
        for kw, f in zip(jobs, futs):
            r = f.result()
            print("  done Phi0=%.2f pi  t_last=%.0f Ek_last=%.2e jmax/jeq=%.2f" %
                  (kw["Phi0"] / np.pi, r["t"][-1], r["Ek"][-1], r["jmax"][-1] / r["jeq"]), flush=True)
            r["job"] = {k: float(v) for k, v in kw.items()}
            out.append(r)
            json.dump(out, open(os.path.join(OUT, "kink_refine2.json"), "w"))


def stage_conv3d():
    """Separate the effects of perpendicular resolution N, axial resolution Nc and
    viscosity nu on the kink at 7 pi (baseline N=64, Nc=32, nu=2e-4: t_nl ~ 52)."""
    from concurrent.futures import ProcessPoolExecutor
    base = dict(Phi0=7 * np.pi, t_end=170.0, p_prof=3)
    jobs = [dict(base, N=96, Nc=32), dict(base, N=64, Nc=48), dict(base, N=128, Nc=32),
            dict(base, N=64, Nc=32, nu=1.0e-3)]
    out = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(K.scan_job, **kw) for kw in jobs]
        for kw, f in zip(jobs, futs):
            r = f.result()
            print("  done N=%d Nc=%d nu=%g  t_last=%.0f Ek_last=%.2e jmax/jeq=%.2f" %
                  (kw.get("N", 64), kw.get("Nc", 32), kw.get("nu", 2e-4), r["t"][-1],
                   r["Ek"][-1], r["jmax"][-1] / r["jeq"]), flush=True)
            r["job"] = {k: float(v) for k, v in kw.items()}
            out.append(r)
            json.dump(out, open(os.path.join(OUT, "kink_conv3d.json"), "w"))


def stage_conv3d_b():
    """Is the kink real?  At N=128, nu=2e-4 the 7 pi loop did NOT go unstable by t=170
    (N=64: t_nl=52, N=96: 150).  Separate 'resolved viscous damping of fine structure'
    from 'low-resolution artefact' with lower viscosity and a much stronger twist."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [dict(Phi0=7 * np.pi, t_end=170.0, p_prof=3, N=128, Nc=32, nu=5.0e-5),
            dict(Phi0=7 * np.pi, t_end=170.0, p_prof=3, N=128, Nc=32, nu=1.0e-4),
            dict(Phi0=10 * np.pi, t_end=130.0, p_prof=3, N=128, Nc=32, nu=2.0e-4)]
    out = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(K.scan_job, **kw) for kw in jobs]
        for kw, f in zip(jobs, futs):
            r = f.result()
            print("  done Phi0=%.0fpi N=%d nu=%g  t_last=%.0f Ek_last=%.2e jmax/jeq=%.2f" %
                  (kw["Phi0"] / np.pi, kw["N"], kw["nu"], r["t"][-1], r["Ek"][-1],
                   r["jmax"][-1] / r["jeq"]), flush=True)
            r["job"] = {k: float(v) for k, v in kw.items()}
            out.append(r)
            json.dump(out, open(os.path.join(OUT, "kink_conv3d_b.json"), "w"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "prod"
    if what == "conv3d_b":
        stage_conv3d_b()
    if what == "conv3d":
        stage_conv3d()
    if what == "prod":
        run_prod()
    if what == "prod128":
        run_prod(Phi0=10 * np.pi, t_end=120.0, N=128, Nc=32, nu=2.0e-4, tag="_N128_10pi")
    elif what == "refine":
        stage_refine()
    elif what == "refine2":
        stage_refine2()
