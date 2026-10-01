#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
3D line-tied kink instability of a twisted coronal loop with ZERO NET CURRENT
=============================================================================

A 3D reduced-MHD (RMHD) model of a straightened coronal loop whose footpoints are
anchored in the (dense, immovable) photosphere, built on the same spectral machinery
as the 2D reconnection code and following the physical set-up of the Lare3d kink
studies (Hood, Browning & Van der Linden 2009; Bareford, Hood & Browning 2013).

Physical model
--------------
Long thin flux tube along z (length L, radius a=1) in a strong uniform axial field
B0.  In the RMHD ordering B_perp/B0 ~ a/L [Strauss 1976; Kadomtsev & Pogutse 1974]

    B = B0 z_hat + z_hat x grad(psi),     v = z_hat x grad(phi)

    d(psi)/dt = -[phi,psi] + B0 d(phi)/dz + eta(j) j          (induction / Ohm)
    d(w)/dt   = -[phi,w] + [psi,j] + B0 d(j)/dz + nu lap(w)    (vorticity)

with j = lap_perp(psi) (parallel current), w = lap_perp(phi) (parallel vorticity),
[f,g] = f_x g_y - f_y g_x.  Equivalently  dpsi/dt = B.grad(phi) + eta j  and
dw/dt = -v.grad(w) + B.grad(j) + ...: line bending (B0 d/dz) is the restoring force
that line-tying makes effective, and the Lorentz force B.grad(j) drives the kink.

Line-tying.  The footpoints sit on the planes z=0 and z=L and cannot move:
phi = 0 there (v_perp = 0), exactly the Lare3d condition "velocity components are
zero at the boundaries" (Bareford et al. 2013).  The model conserves energy
E = (1/2) int (|grad psi|^2 + |grad phi|^2) dV exactly in the ideal limit because
d/dt(E_mag+E_kin) = -int B.grad(phi j) dV = -[B0 phi j]_{z=0}^{z=L} = 0.

Numerics.  Fourier pseudo-spectral in (x,y) (2/3 de-aliasing, integrating-factor RK4
for the constant viscosity), and a STAGGERED second-order finite-difference grid in z
(psi, j at cell centres; phi, w at vertices incl. the footpoint planes).  On this
grid the two wave terms B0 d/dz are exact discrete adjoints (summation by parts), so
line-tied Alfven waves reflect with the right frequency and no boundary closure is
needed.  Staggering is also the idea behind Lare3d's grid [Arber et al. 2001].

Resistivity follows Lare3d's kink runs: eta = eta_b + eta_c * H(|j| - j_crit),
ideal until current sheets thin below the resolved scale, then a localized
'anomalous' resistivity switches on (Bareford et al. 2013; Cozzo et al. 2026 Eq. 1;
Hood et al. 2009).

Equilibrium (zero net current).  Family  B_theta(r) = lambda r (1 - r^{2p})^2 for r<1, 0 outside
  => psi(s) = (lambda/2) [ s - 2 s^{p+1}/(p+1) + s^{2p+1}/(2p+1) ],  s = r^2,
  j(r) = 2 lambda (1-s^p)[(1-s^p) - 2p s^p]   (core current + return-current shell),
  I(r<1) = 2 pi r B_theta |_{r=1} = 0          (zero net current),
  twist angle between the footpoints  Phi(r) = L B_theta/(r B0) = Phi0 (1-s^p)^2.
p=1 concentrates the twist at the axis and was stable up to Phi0 = 14 pi (the return-current
shell stabilises it); p=3 (used for all kink results) has a broad twist plateau.
Any axisymmetric j(r) is an RMHD force-free equilibrium because [psi, j(psi)] = 0.

Known limitations (see README_scalings_and_kink.md): reduced MHD only (no compressibility,
parallel flow, pressure, gravity, shock viscosity; Lare3d's Bessel loops have B_theta/B_z ~ 0.7,
outside the RMHD ordering); the kink onset time and threshold are NOT converged in the
perpendicular resolution N (axial resolution N_z is converged) and are damped by viscosity
(nu = 1e-3 stabilises Phi0 = 6 pi); end-state quantities (energy release, helicity,
relaxed radius) are the robust results.

References
----------
 Strauss 1976, Phys. Fluids 19, 134;  Kadomtsev & Pogutse 1974, Sov. Phys. JETP 38, 283.
 Hood & Priest 1979, Solar Phys. 64, 303 (line-tied kink; Phi_c ~ 2.5 pi uniform twist).
 Hood, Browning & Van der Linden 2009, A&A 506, 913 (zero-net-current loops, Lare3d).
 Browning, Gerrard, Hood, Kevis & Van der Linden 2008, A&A 485, 837.
 Bareford, Hood & Browning 2013, A&A 550, A40 (L=20 R_b, line-tied, anomalous eta).
 Gerrard & Hood 2003, Solar Phys. 214, 151;  Baty 2000, A&A 360, 345.
 Arber, Longbottom, Gerrard & Milne 2001, J. Comput. Phys. 171, 151 (Lare3d).
 Taylor 1974, Phys. Rev. Lett. 33, 1139 (helicity-conserving relaxation).
"""
from __future__ import annotations

import os
os.environ.setdefault("MHD_FFT_WORKERS", "4")

import sys
import time
import json
import numpy as np
import scipy.fft as sfft

_WK = int(os.environ.get("MHD_FFT_WORKERS", "4"))


def _rfft2(a):
    return sfft.rfft2(a, axes=(-2, -1), workers=_WK)


def _irfft2(ah, N):
    return sfft.irfft2(ah, s=(N, N), axes=(-2, -1), workers=_WK)


# ============================================================================
class KinkRMHD3D:
    """Line-tied 3D reduced-MHD solver (spectral in x,y; staggered FD in z)."""

    def __init__(self, N=64, W=5.0, L=20.0, Nc=32, B0=1.0, Phi0=6 * np.pi,
                 nu=3.0e-4, eta_b=0.0, eta_c=0.0, j_crit=4.0, dj=0.8,
                 eps=0.01, seed_k=None, cfl=0.6, dt_max=0.4, seed=True, p_prof=1):
        self.N, self.W, self.L, self.Nc = N, W, L, Nc
        self.B0, self.Phi0 = B0, Phi0
        self.nu, self.eta_b, self.eta_c = nu, eta_b, eta_c
        self.j_crit, self.dj = j_crit, dj
        self.cfl, self.dt_max = cfl, dt_max
        self.dx, self.dz = W / N, L / Nc

        x = (np.arange(N) - N // 2) * self.dx           # tube axis at the origin
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        ky = 2 * np.pi * np.fft.rfftfreq(N, d=self.dx)
        self.kx, self.ky = np.meshgrid(kx, ky, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2inv = np.zeros_like(self.k2)
        self.k2inv[self.k2 > 0] = 1.0 / self.k2[self.k2 > 0]
        cut = (2.0 / 3.0) * np.abs(kx).max()
        self.dealias = ((np.abs(self.kx) <= cut) & (np.abs(self.ky) <= cut))
        # Parseval weights for the half-spectrum
        wt = np.full(self.k2.shape, 2.0)
        wt[:, 0] = 1.0
        if N % 2 == 0:
            wt[:, -1] = 1.0
        self.wt = wt
        self.zc = (np.arange(Nc) + 0.5) * self.dz
        self.zv = np.arange(Nc + 1) * self.dz

        # ---- zero-net-current twisted equilibrium -------------------------
        lam = Phi0 / L
        self.lam = lam
        self.p_prof = p = p_prof
        s = np.minimum((self.X**2 + self.Y**2) / 1.0, 1.0)
        # B_theta = lam r (1 - s^p)^2  ->  psi(s) = (lam/2) int_0^s (1-u^p)^2 du
        psi_eq = 0.5 * lam * (s - 2.0 * s**(p + 1) / (p + 1) + s**(2 * p + 1) / (2 * p + 1))
        ph = _rfft2(psi_eq)
        ph[0, 0] = 0.0
        self.psi_eq_hat = ph * self.dealias
        self.psi_hat = np.repeat(self.psi_eq_hat[None], Nc, axis=0).astype(complex)
        self.w_hat = np.zeros((Nc + 1,) + self.k2.shape, complex)
        j_eq = _irfft2(-self.k2 * self.psi_eq_hat, N)
        self.jeq_max = float(np.abs(j_eq).max())

        if seed:
            k = lam if seed_k is None else seed_k
            r2 = self.X**2 + self.Y**2
            env = np.sin(np.pi * self.zv / L)[:, None, None]
            phi0 = (eps * np.exp(-4.0 * r2**2)[None] * env *
                    (self.X[None] * np.sin(k * self.zv)[:, None, None]
                     - self.Y[None] * np.cos(k * self.zv)[:, None, None]))
            w = -self.k2 * _rfft2(phi0) * self.dealias
            w[0] = 0.0
            w[-1] = 0.0
            self.w_hat = w

        self.t = 0.0
        self.Q_ohm = 0.0
        self.Q_visc = 0.0
        self.dt = None
        self._factors_dt = None

    # ------------------------------------------------------------------ RHS
    def _eta_field(self, j):
        if self.eta_c > 0.0:
            ramp = 0.5 * (1.0 + np.tanh((np.abs(j) - self.j_crit) / self.dj))
            return self.eta_b + self.eta_c * ramp
        return self.eta_b

    def _rhs(self, psi_hat, w_hat, want_diag=False):
        N, dz, B0 = self.N, self.dz, self.B0
        kx, ky = self.kx, self.ky
        dea = self.dealias
        phi_hat = -w_hat * self.k2inv
        j_hat = -self.k2 * psi_hat
        ikx_p, iky_p = 1j * kx * psi_hat, 1j * ky * psi_hat
        ikx_j, iky_j = 1j * kx * j_hat, 1j * ky * j_hat
        psx, psy = _irfft2(ikx_p, N), _irfft2(iky_p, N)
        jx, jy = _irfft2(ikx_j, N), _irfft2(iky_j, N)
        j = _irfft2(j_hat, N)
        phx, phy = _irfft2(1j * kx * phi_hat, N), _irfft2(1j * ky * phi_hat, N)
        wx, wy = _irfft2(1j * kx * w_hat, N), _irfft2(1j * ky * w_hat, N)

        phxc, phyc = 0.5 * (phx[:-1] + phx[1:]), 0.5 * (phy[:-1] + phy[1:])
        br1 = phxc * psy - phyc * psx                      # [phi, psi]   (cells)
        br2 = phx[1:-1] * wy[1:-1] - phy[1:-1] * wx[1:-1]  # [phi, w]     (interior vertices)
        psxv, psyv = 0.5 * (psx[:-1] + psx[1:]), 0.5 * (psy[:-1] + psy[1:])
        jxv, jyv = 0.5 * (jx[:-1] + jx[1:]), 0.5 * (jy[:-1] + jy[1:])
        br3 = psxv * jyv - psyv * jxv                      # [psi, j]     (interior vertices)

        dpsi = -_rfft2(br1) * dea + B0 * (phi_hat[1:] - phi_hat[:-1]) / dz
        eta = self._eta_field(j)
        if self.eta_c > 0.0:
            dpsi = dpsi + _rfft2(eta * j) * dea
        dw = np.zeros_like(w_hat)
        dw[1:-1] = ((-_rfft2(br2) + _rfft2(br3)) * dea
                    + B0 * (j_hat[1:] - j_hat[:-1]) / dz)
        if want_diag:
            w = _irfft2(w_hat, N)
            return dpsi, dw, (j, w, eta, psx, psy, phx, phy)
        return dpsi, dw, None

    # ----------------------------------------------------------- stepping
    def _build_factors(self, dt):
        self.dt = dt
        self._factors_dt = dt
        self.Ep = np.exp(-self.eta_b * self.k2 * dt)
        self.Ew = np.exp(-self.nu * self.k2 * dt)
        self.Ep2 = np.exp(-self.eta_b * self.k2 * dt / 2)
        self.Ew2 = np.exp(-self.nu * self.k2 * dt / 2)

    def _bw(self, w):
        w[0] = 0.0
        w[-1] = 0.0
        return w

    def step(self):
        dt = self.dt
        p, w = self.psi_hat, self.w_hat
        Ep, Ew, Ep2, Ew2 = self.Ep, self.Ew, self.Ep2, self.Ew2
        k1p, k1w, diag = self._rhs(p, w, want_diag=True)
        k1p, k1w = dt * k1p, dt * k1w
        k2p, k2w, _ = self._rhs(Ep2 * (p + 0.5 * k1p), self._bw(Ew2 * (w + 0.5 * k1w)))
        k2p, k2w = dt * k2p, dt * k2w
        k3p, k3w, _ = self._rhs(Ep2 * p + 0.5 * k2p, self._bw(Ew2 * w + 0.5 * k2w))
        k3p, k3w = dt * k3p, dt * k3w
        k4p, k4w, _ = self._rhs(Ep * p + Ep2 * k3p, self._bw(Ew * w + Ew2 * k3w))
        k4p, k4w = dt * k4p, dt * k4w
        self.psi_hat = Ep * p + (Ep * k1p + 2 * Ep2 * (k2p + k3p) + k4p) / 6.0
        self.w_hat = self._bw(Ew * w + (Ew * k1w + 2 * Ew2 * (k2w + k3w) + k4w) / 6.0)
        # heating accumulated from the state at the start of the step
        j, wr, eta = diag[0], diag[1], diag[2]
        dV = self.dx**2 * self.dz
        self.Q_ohm += float(np.sum(eta * j**2) * dV) * dt
        wc = 0.5 * (wr[:-1] + wr[1:])
        self.Q_visc += float(self.nu * np.sum(wc**2) * dV) * dt
        self.t += dt
        return diag

    # ---------------------------------------------------------- diagnostics
    def energies(self):
        pref = self.W**2 / self.N**4
        Em = 0.5 * self.dz * pref * float(np.sum(self.wt * self.k2 * np.abs(self.psi_hat)**2))
        Ek = 0.5 * self.dz * pref * float(np.sum(self.wt * self.k2 * np.abs(self.w_hat * self.k2inv)**2))
        return Em, Ek

    def max_signal_speed(self, diag=None):
        N = self.N
        psx = _irfft2(1j * self.kx * self.psi_hat, N)
        psy = _irfft2(1j * self.ky * self.psi_hat, N)
        bperp = float(np.sqrt(psx**2 + psy**2).max())
        phi_hat = -self.w_hat * self.k2inv
        phx = _irfft2(1j * self.kx * phi_hat, N)
        phy = _irfft2(1j * self.ky * phi_hat, N)
        vperp = float(np.sqrt(phx**2 + phy**2).max())
        return bperp + vperp

    def choose_dt(self):
        vmax = max(self.max_signal_speed(), 1e-3)
        dt = min(self.cfl * self.dx / vmax, 0.9 * self.dz / self.B0, self.dt_max)
        return dt

    def axis_position(self, plane=None):
        """(x,y) of the tube axis on a z-plane, sub-grid refined.

        For B_theta > 0 the flux function psi increases outward from the axis and
        plateaus outside the tube (zero net current), so the axis is the MINIMUM of
        psi (the O-point of the field)."""
        plane = self.Nc // 2 if plane is None else plane
        psi = _irfft2(self.psi_hat[plane], self.N)
        i, j = np.unravel_index(np.argmin(psi), psi.shape)

        def refine(fm, f0, fp, x0):
            den = fm - 2 * f0 + fp
            return x0 + (0.5 * (fm - fp) / den) * self.dx if den != 0 else x0
        N = self.N
        xs = refine(psi[(i - 1) % N, j], psi[i, j], psi[(i + 1) % N, j], self.X[i, j])
        ys = refine(psi[i, (j - 1) % N], psi[i, j], psi[i, (j + 1) % N], self.Y[i, j])
        return float(xs), float(ys)

    def current_field(self):
        return _irfft2(-self.k2 * self.psi_hat, self.N)

    def psi_real(self):
        return _irfft2(self.psi_hat, self.N)

    def run(self, t_end, record_dt=1.0, snap_times=(), verbose=True, stop_if=None):
        rec = {k: [] for k in ("t", "Em", "Ek", "Qo", "Qv", "jmax", "xa", "ya", "wall")}
        snaps = {}
        next_rec = 0.0
        snap_times = sorted(snap_times)
        si = 0
        t0 = time.time()
        step_n = 0
        self._build_factors(self.choose_dt())
        jmax_last = self.jeq_max
        while self.t < t_end - 1e-9:
            if step_n % 10 == 0:
                dt_new = self.choose_dt()
                if abs(dt_new - self._factors_dt) > 0.05 * self._factors_dt:
                    self._build_factors(dt_new)
            if self.t + self.dt > t_end:
                self._build_factors(t_end - self.t)
            if si < len(snap_times) and self.t + self.dt >= snap_times[si]:
                self._build_factors(max(snap_times[si] - self.t, 1e-6)) \
                    if snap_times[si] > self.t + 1e-6 else None
            diag = self.step()
            step_n += 1
            jmax_last = float(np.abs(diag[0]).max())
            if self.t >= next_rec - 1e-9:
                Em, Ek = self.energies()
                xa, ya = self.axis_position()
                for k, v in zip(("t", "Em", "Ek", "Qo", "Qv", "jmax", "xa", "ya", "wall"),
                                (self.t, Em, Ek, self.Q_ohm, self.Q_visc, jmax_last, xa, ya,
                                 time.time() - t0)):
                    rec[k].append(v)
                next_rec += record_dt
                if verbose and len(rec["t"]) % 10 == 0:
                    print("  t=%7.2f/%g  Em=%.5f Ek=%.3e jmax=%.2f Qo=%.2e  [%.0fs]" %
                          (self.t, t_end, Em, Ek, jmax_last, self.Q_ohm,
                           time.time() - t0), flush=True)
            while si < len(snap_times) and self.t >= snap_times[si] - 1e-6:
                jf = self.current_field()
                wv = _irfft2(self.w_hat, self.N)
                wc = 0.5 * (wv[:-1] + wv[1:])
                H = self._eta_field(jf) * jf**2 + self.nu * wc**2   # heating rate density
                snaps[snap_times[si]] = dict(
                    t=self.t, psi=self.psi_real().astype(np.float32),
                    j=jf.astype(np.float32), H=np.asarray(H, np.float32))
                si += 1
            if not np.isfinite(self.psi_hat).all():
                print("  non-finite state at t=%.2f" % self.t)
                break
            if stop_if is not None and stop_if(self, rec):
                break
        out = {k: np.array(v) for k, v in rec.items()}
        out["snaps"] = snaps
        return out


# ============================================================================
# Verification tests of the solver
# ============================================================================
def test_alfven_wave(N=32, Nc=32, L=20.0, T_periods=1.5):
    """Line-tied Alfven wave in a uniform field: fundamental period 2L/B0."""
    s = KinkRMHD3D(N=N, W=5.0, L=L, Nc=Nc, Phi0=0.0, nu=0.0, seed=False)
    k = 2 * np.pi / s.W
    psi0 = 1e-4 * np.cos(k * s.X)[None] * np.cos(np.pi * s.zc / L)[:, None, None]
    s.psi_hat = _rfft2(psi0) * s.dealias
    s.psi_hat[:, 0, 0] = 0.0
    s._build_factors(0.4)
    ts, Es = [], []
    while s.t < T_periods * 2 * L:
        s.step()
        Em, Ek = s.energies()
        ts.append(s.t)
        Es.append(Em)
    ts, Es = np.array(ts), np.array(Es)
    # E_mag(t) = E0 cos^2(omega t) -> minima at omega t = pi/2: period T = 2L/B0
    from scipy.signal import argrelextrema
    mins = argrelextrema(Es, np.less)[0]
    t_min = ts[mins]
    period = 2 * np.mean(np.diff(t_min)) if len(t_min) > 1 else np.nan
    # residual energy conservation
    Em, Ek = s.energies()
    return period, 2 * L / s.B0, (Es + 0 * Es).max() / Es[0]


def test_equilibrium_static(Phi0=3 * np.pi, tend=30.0):
    s = KinkRMHD3D(N=48, W=5.0, L=20.0, Nc=24, Phi0=Phi0, nu=1e-4, seed=False)
    Em0, _ = s.energies()
    r = s.run(tend, record_dt=5.0, verbose=False)
    return r["Ek"].max(), (r["Em"][-1] - Em0) / Em0


# ============================================================================
# Field-line tracing and field-line sampling of the heating  (Reid et al. 2021)
# ============================================================================
def field_components(psi, W):
    """B_perp = z_hat x grad(psi) on every z-plane of a real (Nc,N,N) array."""
    N = psi.shape[-1]
    kx = 2 * np.pi * np.fft.fftfreq(N, d=W / N)
    ky = 2 * np.pi * np.fft.rfftfreq(N, d=W / N)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    ph = _rfft2(psi.astype(float))
    psx = _irfft2(1j * KX * ph, N)
    psy = _irfft2(1j * KY * ph, N)
    return -psy, psx


def trace_lines(Bx, By, W, L, B0, x0, y0):
    """Trace field lines from the z=0 footpoints (x0,y0) to z=L:
    dx/dz = B_x/B0,  dy/dz = B_y/B0   (RMHD field lines), midpoint RK2.
    Returns path of shape (2*Nc+1, 2, M); index 2k+1 sits at the cell-centre
    plane z=(k+1/2)dz."""
    from scipy.ndimage import map_coordinates
    Nc, N, _ = Bx.shape
    dx, dz = W / N, L / Nc
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    y = np.atleast_1d(np.asarray(y0, float)).copy()
    nst = 2 * Nc
    h = L / nst
    path = np.zeros((nst + 1, 2, x.size))
    path[0, 0], path[0, 1] = x, y

    def fld(xq, yq, z):
        kz = np.clip(z / dz - 0.5, 0.0, Nc - 1.000001)
        k0 = int(kz)
        f = kz - k0
        ix = xq / dx + N // 2
        iy = yq / dx + N // 2
        out = []
        for B in (Bx, By):
            a = map_coordinates(B[k0], [ix, iy], order=1, mode="wrap")
            b = map_coordinates(B[k0 + 1], [ix, iy], order=1, mode="wrap")
            out.append(((1 - f) * a + f * b) / B0)
        return out

    z = 0.0
    for i in range(nst):
        bx1, by1 = fld(x, y, z)
        bx2, by2 = fld(x + 0.5 * h * bx1, y + 0.5 * h * by1, z + 0.5 * h)
        x, y, z = x + h * bx2, y + h * by2, z + h
        path[i + 1, 0], path[i + 1, 1] = x, y
    return path


def sample_along(path, Hsnap, W):
    """Sample a (Nc,N,N) field along traced lines at the cell-centre planes."""
    from scipy.ndimage import map_coordinates
    Nc, N, _ = Hsnap.shape
    dx = W / N
    M = path.shape[2]
    out = np.zeros((Nc, M))
    for k in range(Nc):
        ix = path[2 * k + 1, 0] / dx + N // 2
        iy = path[2 * k + 1, 1] / dx + N // 2
        out[k] = map_coordinates(Hsnap[k].astype(float), [ix, iy], order=1, mode="wrap")
    return out


def physical_scales(B_G=10.0, n_m3=1.0e15, a_Mm=4.5):
    """Map code units (a=1, v_A=1, B0=1, mu0=1) to SI.

    Defaults are the reference values of Reid, Cargill, Johnston & Hood (2021):
    B = 10 G, n = 1e15 m^-3 (v_A ~ 690 km/s), strand radius 4.5 Mm, so L = 20 a = 90 Mm
    -- the same aspect ratio L/R_b = 20 as the Lare3d kink loops of Bareford et al. (2013).
    Uses the standard textbook formulae (identical to what PlasmaPy's formulary
    returns; see coronal_parameters() in mhd_reconnection_coronal_heating.py)."""
    mu0, m_p = 1.25663706212e-6, 1.67262192369e-27
    B0 = B_G * 1e-4
    vA = B0 / np.sqrt(mu0 * n_m3 * m_p)
    a = a_Mm * 1e6
    tauA = a / vA
    return dict(B0=B0, n=n_m3, vA=vA, a=a, tauA=tauA,
                Q_unit=(B0**2 / mu0) * vA / a,        # W m^-3 per unit code heating rate
                E_unit=(B0**2 / mu0) * a**3)          # J per unit code energy


class KinkHeatingDriver:
    """Q(s,t) for the 1D loop from the 3D kink heating sampled along a field line
    (the Reid et al. 2021 prescription), with code->SI scaling."""

    def __init__(self, loop, t_code, Hline, scales, L_code):
        self.t_phys = np.asarray(t_code) * scales["tauA"]
        self.Hline = np.asarray(Hline)                      # (nt, Nc) code units
        Qn = self.Hline * scales["Q_unit"]
        scor = loop.s[loop.corona]
        zq = (scor - scor.min()) / (scor.max() - scor.min()) * L_code
        zc = (np.arange(Qn.shape[1]) + 0.5) * L_code / Qn.shape[1]
        self.Qcor = np.array([np.interp(zq, zc, q) for q in Qn])
        self.loop = loop

    def Q_of(self, t):
        Q = np.zeros(self.loop.N)
        if t < self.t_phys[0] or t > self.t_phys[-1]:
            return Q
        k = int(np.clip(np.searchsorted(self.t_phys, t) - 1, 0, len(self.t_phys) - 2))
        f = (t - self.t_phys[k]) / (self.t_phys[k + 1] - self.t_phys[k])
        Q[self.loop.corona] = (1 - f) * self.Qcor[k] + f * self.Qcor[k + 1]
        return Q


def production_run(Phi0, t_end, N=80, Nc=32, W=5.0, L=20.0, nu=1.0e-3, eta_c=1.0e-3,
                   j_crit=None, snap_dt=2.0, eps=0.01, verbose=True):
    """Nonlinear kink run with Lare3d-style anomalous resistivity."""
    s = KinkRMHD3D(N=N, W=W, L=L, Nc=Nc, Phi0=Phi0, nu=nu, eta_b=0.0,
                   eta_c=eta_c, j_crit=1.0, eps=eps)
    if j_crit is None:
        s.j_crit = 1.8 * s.jeq_max       # 'significantly higher than the initial maximum'
        s.dj = 0.08 * s.j_crit
    else:
        s.j_crit = j_crit
    snap_times = list(np.arange(0.0, t_end + 1e-9, snap_dt))
    r = s.run(t_end, record_dt=1.0, snap_times=snap_times, verbose=verbose)
    r["meta"] = dict(Phi0=Phi0, N=N, Nc=Nc, W=W, L=L, nu=nu, eta_c=eta_c,
                     j_crit=s.j_crit, jeq=s.jeq_max, lam=s.lam)
    return s, r


def scan_job(Phi0, t_end=200.0, N=64, Nc=32, W=5.0, L=20.0, nu=2.0e-4, eps=0.01,
             jstop=4.0, p_prof=1):
    """Ideal (eta=0) linear-stability run for one twist value; stops once the
    current has grown to jstop * (equilibrium maximum) i.e. at nonlinear onset."""
    s = KinkRMHD3D(N=N, W=W, L=L, Nc=Nc, Phi0=Phi0, nu=nu, eps=eps, p_prof=p_prof)
    stop = lambda sim, rec: rec["jmax"][-1] > jstop * sim.jeq_max if rec["jmax"] else False
    r = s.run(t_end, record_dt=2.0, verbose=False, stop_if=stop)
    r.pop("snaps", None)
    r.update(Phi0=Phi0, lam=s.lam, jeq=s.jeq_max, N=N, Nc=Nc, W=W, L=L, nu=nu,
             p_prof=p_prof)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def stage_scan(Phis, t_end=200.0, workers=3, fname="kink_scan.json", **kw):
    from concurrent.futures import ProcessPoolExecutor
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "scaling_results"), exist_ok=True)
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(scan_job, p, t_end, **kw) for p in Phis]
        for p, f in zip(Phis, futs):
            r = f.result()
            print("  Phi0=%.2f pi done: t_last=%.0f  Ek_last=%.2e  jmax/jeq=%.2f" %
                  (p / np.pi, r["t"][-1], r["Ek"][-1], r["jmax"][-1] / r["jeq"]), flush=True)
            out.append(r)
            json.dump(out, open(os.path.join(here, "scaling_results", fname), "w"))
    return out


if __name__ == "__main__":
    if "--scan" in sys.argv:
        Phis = [m * np.pi for m in (8, 9, 10, 11, 12, 14)]
        stage_scan(Phis)
    if "--scan-p3" in sys.argv:
        Phis = [m * np.pi for m in (4, 5, 6, 7, 8, 9)]
        stage_scan(Phis, fname="kink_scan_p3.json", p_prof=3)
    if "--test" in sys.argv:
        t0 = time.time()
        per, expect, _ = test_alfven_wave()
        print("Alfven wave period: measured %.3f  expected 2L/B0 = %.3f  (err %.2f%%)" %
              (per, expect, 100 * (per - expect) / expect))
        ek, dEm = test_equilibrium_static()
        print("Static equilibrium (no seed): max Ek = %.2e,  rel dEm = %.2e" % (ek, dEm))
        print("tests %.0fs" % (time.time() - t0))
