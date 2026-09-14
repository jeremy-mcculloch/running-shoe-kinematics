"""Shared exponential-memory machinery for Prony / sum-of-exponentials models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn


def stable_one_minus_exp_over_x(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute ``(1 - exp(-x)) / x`` stably for small and large ``x``.

    As ``x -> 0`` the limit is 1. As ``x -> +inf`` the value decays as ``1/x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    small = np.abs(x_arr) < 1.0e-8
    # Series: 1 - x/2 + x^2/6 - ...
    xs = x_arr[small]
    out[small] = 1.0 - 0.5 * xs + (xs * xs) / 6.0
    large = ~small
    xl = x_arr[large]
    out[large] = -np.expm1(-xl) / xl
    return out


def prony_inverse_step(
    *,
    FVE: NDArray[np.float64],
    FVE_prev: NDArray[np.float64],
    FE_prev: NDArray[np.float64],
    q: NDArray[np.float64],
    g_inf: float,
    h: NDArray[np.float64],
    tau: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One online inverse step for a Prony relaxation solid.

    Model (componentwise)::

        F_VE = G * dF_e/dt
        G(t) = g_inf + sum_k h_k exp(-t/tau_k)

    with internal states ``q_k = int exp(-(t-s)/tau_k) dF_e(s)``.

    Parameters
    ----------
    FVE, FVE_prev, FE_prev
        Shape ``(n_components,)``.
    q
        Shape ``(n_modes, n_components)``, updated in place conceptually via return.
    h, tau
        Shape ``(n_modes,)``.

    Returns
    -------
    FE, q_new
    """
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    n_modes = h.size
    n_comp = FVE.size
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (n_modes, n_comp):
        raise ValueError(f"q must have shape {(n_modes, n_comp)}, got {q.shape}")

    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    if dt == 0.0:
        # No hereditary advance: invert the instantaneous modulus G(0)=1.
        # With q frozen, F_VE = g_inf*FE + sum h_k q_k  is not generally
        # consistent with an instantaneous force jump under G(0)=1.
        # Use the glassy response F_e jump = F_VE jump when dt=0.
        dFVE = FVE - FVE_prev
        FE = FE_prev + dFVE
        return FE, q.copy()

    alpha = np.exp(-dt / tau)  # (n_modes,)
    beta = stable_one_minus_exp_over_x(dt / tau)  # (1-e^{-x})/x
    # q_k^i = alpha_k q_k^{i-1} + beta_k (FE^i - FE^{i-1})
    sum_h_beta = float(np.dot(h, beta))
    denom = g_inf + sum_h_beta
    if denom <= 0.0:
        raise RuntimeError(f"non-positive Prony inverse denominator {denom}")

    # FVE^i = (g_inf + sum h beta) FE^i + sum h alpha q^{i-1} - (sum h beta) FE^{i-1}
    filtered = np.einsum("m,mc->c", h * alpha, q)
    FE = (FVE - filtered + sum_h_beta * FE_prev) / denom

    dFE = FE - FE_prev
    q_new = alpha[:, None] * q + beta[:, None] * dFE
    return FE, q_new


def prony_forward_step(
    *,
    FE: NDArray[np.float64],
    FE_prev: NDArray[np.float64],
    q: NDArray[np.float64],
    g_inf: float,
    h: NDArray[np.float64],
    tau: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Forward Prony step ``F_VE = G * dF_e`` (for validation)."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    n_modes = h.size
    n_comp = FE.size
    q = np.asarray(q, dtype=np.float64)
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if dt == 0.0:
        dFE = FE - FE_prev
        FVE = g_inf * FE + np.einsum("m,mc->c", h, q) + np.einsum("m,c->c", h, dFE) * 0.0
        # Instantaneous: G(0)=1 => dFVE = dFE, and FVE = FE if starting from zero states
        # consistently: FVE = g_inf*FE + sum h q, with q jumping by dFE for glassy part.
        # Use FVE = FE_prev_mapped + dFE with G(0)=1:
        FVE_inst = FE_prev * 0.0  # placeholder overwritten below
        FVE_inst = g_inf * FE_prev + np.einsum("m,mc->c", h, q) + dFE
        return FVE_inst, q.copy()

    alpha = np.exp(-dt / tau)
    beta = stable_one_minus_exp_over_x(dt / tau)
    dFE = FE - FE_prev
    q_new = alpha[:, None] * q + beta[:, None] * dFE
    FVE = g_inf * FE + np.einsum("m,mc->c", h, q_new)
    return FVE, q_new


def logspaced_taus(tau_min: float, tau_max: float, num_modes: int) -> NDArray[np.float64]:
    if not (tau_min > 0.0 and tau_max > tau_min):
        raise ValueError("require 0 < tau_min < tau_max")
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    return np.geomspace(tau_min, tau_max, int(num_modes), dtype=np.float64)


def fractional_kernel_weights(
    alpha: float,
    taus: NDArray[np.float64],
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    n_collocation: int = 80,
) -> NDArray[np.float64]:
    """Nonnegative least-squares weights for ``t^{alpha-1} ≈ Σ a_k e^{-t/τ_k}``.

    Collocates on a log-spaced time grid covering the approximation window.
    The integral representation

    ``t^{α-1} = 1/Γ(1-α) ∫ τ^{α-1} e^{-t/τ} d(ln τ)``

    motivates the log-spaced poles; NNLS then fits accurate amplitudes on that
    basis over ``[t_min, t_max]``.
    """
    from scipy.optimize import nnls

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    taus = np.asarray(taus, dtype=np.float64).reshape(-1)
    if taus.size < 1:
        raise ValueError("taus must be non-empty")
    if np.any(taus <= 0.0):
        raise ValueError("all tau_k must be positive")

    tau_min = float(np.min(taus))
    tau_max = float(np.max(taus))
    # Collocate inside the pole window (avoid extreme edges).
    if t_min is None:
        t_min = tau_min * 3.0
    if t_max is None:
        t_max = tau_max / 3.0
    if not (t_min > 0.0 and t_max > t_min):
        raise ValueError("require 0 < t_min < t_max for kernel collocation")

    t_colloc = np.geomspace(t_min, t_max, int(max(n_collocation, 2 * taus.size)))
    A = np.exp(-t_colloc[:, None] / taus[None, :])
    b = np.power(t_colloc, alpha - 1.0)
    try:
        weights, _residual = nnls(A, b, maxiter=max(50 * A.shape[1], 2000))
    except RuntimeError:
        weights = np.zeros(taus.size, dtype=np.float64)

    if float(np.sum(weights)) <= 0.0 or not np.all(np.isfinite(weights)):
        # Fallback: truncated integral quadrature in ln τ.
        pref = 1.0 / float(gamma_fn(1.0 - alpha))
        log_tau = np.log(taus)
        if taus.size == 1:
            dx = np.array([1.0], dtype=np.float64)
        else:
            dx = np.empty(taus.size, dtype=np.float64)
            dx[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
            dx[0] = log_tau[1] - log_tau[0]
            dx[-1] = log_tau[-1] - log_tau[-2]
        weights = pref * dx * np.power(taus, alpha - 1.0)
        # Calibrate amplitude at geometric-mean time in the window.
        t_ref = float(np.sqrt(t_min * t_max))
        approx = float(np.dot(weights, np.exp(-t_ref / taus)))
        exact = float(t_ref ** (alpha - 1.0))
        if approx > 0.0:
            weights *= exact / approx
    return np.asarray(weights, dtype=np.float64)


def fung_prony_weights(
    C: float,
    tau1: float,
    tau2: float,
    num_modes: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Logarithmic quadrature of Fung's ``S(τ)=C/τ`` spectrum.

    Returns ``(g_inf, h, tau)`` with ``sum(h) = 1 - g_inf`` and ``G(0)=1``.
    """
    if C <= 0.0:
        raise ValueError("C must be > 0")
    if not (tau1 > 0.0 and tau2 > tau1):
        raise ValueError("require 0 < tau1 < tau2")
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")

    ln_ratio = float(np.log(tau2 / tau1))
    g_inf = 1.0 / (1.0 + C * ln_ratio)
    taus = logspaced_taus(tau1, tau2, num_modes)
    log_tau = np.log(taus)
    if num_modes == 1:
        dx = np.array([ln_ratio], dtype=np.float64)
    else:
        dx = np.empty(num_modes, dtype=np.float64)
        dx[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
        dx[0] = log_tau[1] - log_tau[0]
        dx[-1] = log_tau[-1] - log_tau[-2]
        # Rescale so sum dx = ln(tau2/tau1) exactly (endpoints).
        dx *= ln_ratio / float(np.sum(dx))

    h = g_inf * C * dx
    # Enforce exact normalization sum h = 1 - g_inf
    target = 1.0 - g_inf
    s = float(np.sum(h))
    if s <= 0.0:
        raise RuntimeError("Fung quadrature produced non-positive weights")
    h *= target / s
    return g_inf, h, taus


def integrate_force_driven_exponentials(
    *,
    F: NDArray[np.float64],
    F_prev: NDArray[np.float64],
    q: NDArray[np.float64],
    tau: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Exact-in-interval update of ``q_k = int e^{-(t-s)/tau_k} F(s) ds``.

    Assumes piecewise-linear ``F`` between the previous and current samples.
    """
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64)
    n_modes = tau.size
    n_comp = F.size
    if q.shape != (n_modes, n_comp):
        raise ValueError(f"q must have shape {(n_modes, n_comp)}, got {q.shape}")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if dt == 0.0:
        return q.copy()

    alpha = np.exp(-dt / tau)
    # q^+ = e^{-dt/τ} q + ∫_0^{dt} e^{-(dt-s)/τ} (F_prev + r s) ds
    #     = α q + F_prev τ(1-α) + r (τ dt - τ²(1-α))
    one_m_alpha = -np.expm1(-dt / tau)  # 1 - e^{-dt/τ}
    r = (F - F_prev) / dt
    # term0: F_prev * τ * (1-α)
    # term1: r * (τ*dt - τ²*(1-α)) = r*τ*dt * (1 - (1-α)/(dt/τ))
    #       = r * τ * dt * (1 - stable_one_minus_exp_over_x(dt/τ))
    phi = stable_one_minus_exp_over_x(dt / tau)  # (1-α)/(dt/τ) = τ(1-α)/dt
    # τ(1-α) = dt * phi
    # τ dt - τ²(1-α) = τ dt (1 - τ(1-α)/dt) = τ dt (1 - phi)
    q_new = (
        alpha[:, None] * q
        + (dt * phi)[:, None] * F_prev
        + (tau * dt * (1.0 - phi))[:, None] * r
    )
    return q_new


class ExponentialSpectrum:
    """Log-spaced spectrum container ``(tau_k, weight_k)``."""

    __slots__ = ("tau", "weight", "g_inf")

    def __init__(
        self,
        tau: NDArray[np.float64],
        weight: NDArray[np.float64],
        *,
        g_inf: float = 0.0,
    ) -> None:
        self.tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        self.weight = np.asarray(weight, dtype=np.float64).reshape(-1)
        if self.tau.shape != self.weight.shape:
            raise ValueError("tau and weight must have the same shape")
        if np.any(self.tau <= 0.0):
            raise ValueError("all relaxation times must be positive")
        self.g_inf = float(g_inf)

    @property
    def n_modes(self) -> int:
        return int(self.tau.size)
