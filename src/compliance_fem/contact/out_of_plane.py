"""2.5D out-of-plane tangent stiffness of a converged sagittal contact state.

The sagittal FEM is two-dimensional (plane strain, per metre of width). This
module adds three **auxiliary, linearized** out-of-plane stiffnesses around a
converged contact state, without changing it:

    generalized dofs   dq = [dz, dphi_x, dphi_y]        (m, rad, rad)
    conjugate loads    dp = [dF_z, dM_x, dM_y]          (N, N·m, N·m)

    k_z    = dF_z / dz          lateral (mediolateral) translation   [N/m]
    k_roll = dM_x / dphi_x      roll, about the heel->toe x axis      [N·m/rad]
    k_yaw  = dM_y / dphi_y      yaw, about the vertical y axis        [N·m/rad]

Axes: x heel -> toe, y up, z mediolateral (right-handed). A rotation dphi_y
about y moves a point at horizontal lever r = x - x_yaw by du_z = -r dphi_y; a
rotation dphi_x about x lifts a point at z by du_y = z dphi_x.

Model (a 2.5D strip approximation, not a 3D FEM)
-------------------------------------------------
* Transverse strips: each bottom contact node carries a strip of width b_i;
  every out-of-plane slice z behaves like the 2D plane-strain section, and the
  slices do not interact.
* Active-set tangent: the converged active set A is frozen (small
  perturbations do not change contact), so the values are one-sided local
  tangents that can jump when A changes.
* Normal tangent: K_n = H_AA^{-1} with H_AA the **pure normal** gap
  compliance of the active nodes (mu_eff = 0, symmetric positive definite),
  per metre width: dlambda_A = K_n dd_n,A, positive indentation -> positive
  compressive reaction. H_AA is Cholesky-factored once; no inverse is formed.
  The unaugmented H_AA is the consistent tangent here: the roll indentation
  z dphi_x is antisymmetric across each strip, so it adds no net vertical force
  (the force-control constraint sum(lambda) = -Fy* does not bind) and, at first
  order, no toe generalized force (Q_alpha comes from the in-plane top forces),
  so neither the frame translation T_y nor the toe angle couples in.
* Lateral shear: K_t = eta_t K_n with eta_t the lateral-to-normal elastic
  stiffness ratio (an approximation: the sagittal FEM does not model lateral
  shear). eta_t is not a friction coefficient.

Width weighting (per-metre model; the width enters exactly once)
-----------------------------------------------------------------
For a constant width b the strip integrals give total = b K, roll lever
int z^2 dz = b^3/12. With node widths b_i the symmetric weighting
d^T H_AA^{-1} d uses

    k_z    = eta_t * (sqrt(b))^T       H_AA^{-1} (sqrt(b))
    k_yaw  = eta_t * (sqrt(b) r)^T     H_AA^{-1} (sqrt(b) r)
    k_roll =         (sqrt(b^3/12))^T  H_AA^{-1} (sqrt(b^3/12))

i.e. K_phys = diag(sqrt b) K_n diag(sqrt b) and W_roll = diag(b^2/12). For a
constant width this is exact under the strip assumption
(k_roll = b^2/12 * 1^T K_phys 1); for a varying width it is the geometric-mean
strip approximation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from compliance_fem.contact.contact_operator import ContactOperator
from compliance_fem.contact.continuous_contact import ContactSolution, _normal_operator

OUT_OF_PLANE_MODEL = "2.5D transverse-strip active-set tangent"
SHEAR_MODEL_NORMAL_RATIO = "normal_stiffness_ratio"
YAW_REFERENCE_COP = "center_of_pressure"
YAW_REFERENCE_SHOE = "shoe_reference"
YAW_REFERENCES = (YAW_REFERENCE_COP, YAW_REFERENCE_SHOE)
WIDTH_DISTRIBUTION_ASSUMPTION = (
    "normal tangent stiffness uniform across each transverse strip of width b_i; "
    "slices z do not interact; widths enter as sqrt(b) (translation, yaw) and sqrt(b^3/12) (roll)"
)
FEM_WIDTH_CONVENTION = "plane strain, forces and stiffness per metre of out-of-plane width"
TANGENT_KIND = "unaugmented pure-normal active block H_AA (mu_eff = 0), Cholesky"
OUT_OF_PLANE_DOFS = ("z", "phi_x", "phi_y")
OUT_OF_PLANE_UNITS = {
    "lateral_translation_stiffness": "N/m",
    "roll_stiffness": "N*m/rad",
    "yaw_stiffness": "N*m/rad",
    "yaw_reference_x": "m",
}

STATUS_OK = "ok"
STATUS_NO_CONTACT = "no_contact"
STATUS_ILL_CONDITIONED = "ill_conditioned"
STATUS_SINGULAR = "tangent_singular"
STATUS_NOT_PD = "tangent_not_positive_definite"
STATUS_DISABLED = "disabled"
STATUS_NO_RATIO = "lateral_ratio_unavailable"
STATUS_NO_COP = "yaw_reference_unavailable"

# 1/rcond above this is treated as numerically singular (NaN results).
SINGULAR_CONDITION = 1.0e14


def lateral_ratio_from_poisson(nu: float) -> float:
    """Thin-layer shear-to-constrained-normal modulus ratio G/M = (1 - 2 nu) / (2 (1 - nu))."""
    nu = float(nu)
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}.")
    return (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))


@dataclass(frozen=True)
class OutOfPlaneSettings:
    """2.5D stiffness settings (SI).

    ``shoe_width_m`` is the constant strip width. ``shoe_width_x`` /
    ``shoe_width_values`` (both or neither) give a heel -> toe width profile in
    metres, linearly interpolated at the contact-node reference x (clamped at
    the ends). ``lateral_to_normal_stiffness_ratio = None`` uses the thin-layer
    value from the contact-foam Poisson ratio (an assumption, reported as such).
    ``fixed_yaw_reference_x`` is a material x (default L/2) used by
    ``yaw_reference = "shoe_reference"`` and as the fallback without load.
    """

    enabled: bool = True
    shoe_width_m: float = 0.10
    shoe_width_x: tuple[float, ...] | None = None
    shoe_width_values: tuple[float, ...] | None = None
    out_of_plane_shear_model: str = SHEAR_MODEL_NORMAL_RATIO
    lateral_to_normal_stiffness_ratio: float | None = None
    yaw_reference: str = YAW_REFERENCE_COP
    fixed_yaw_reference_x: float | None = None
    condition_number_warning: float = 1.0e10
    # Flags states whose smallest active reaction (relative to the peak) or smallest
    # inactive gap (relative to the node spacing h) is below this fraction.
    near_change_fraction: float = 0.005

    def __post_init__(self) -> None:
        if not (np.isfinite(self.shoe_width_m) and self.shoe_width_m > 0.0):
            raise ValueError(f"shoe_width_m must be finite and positive, got {self.shoe_width_m!r}.")
        if (self.shoe_width_x is None) != (self.shoe_width_values is None):
            raise ValueError("shoe_width_x and shoe_width_values must be given together.")
        if self.shoe_width_x is not None and self.shoe_width_values is not None:
            x = np.asarray(self.shoe_width_x, dtype=float)
            b = np.asarray(self.shoe_width_values, dtype=float)
            if x.ndim != 1 or x.size < 2 or b.shape != x.shape:
                raise ValueError("shoe_width_x and shoe_width_values must be 1D with the same length >= 2.")
            if not (np.all(np.isfinite(x)) and np.all(np.diff(x) > 0.0)):
                raise ValueError("shoe_width_x must be finite and strictly increasing.")
            if not (np.all(np.isfinite(b)) and np.all(b > 0.0)):
                raise ValueError("shoe_width_values must be finite and positive.")
        if self.out_of_plane_shear_model != SHEAR_MODEL_NORMAL_RATIO:
            raise ValueError(f"out_of_plane_shear_model must be {SHEAR_MODEL_NORMAL_RATIO!r}.")
        eta = self.lateral_to_normal_stiffness_ratio
        if eta is not None and not (np.isfinite(eta) and eta >= 0.0):
            raise ValueError(f"lateral_to_normal_stiffness_ratio must be finite and >= 0, got {eta!r}.")
        if self.yaw_reference not in YAW_REFERENCES:
            raise ValueError(f"yaw_reference must be one of {YAW_REFERENCES}, got {self.yaw_reference!r}.")
        if self.fixed_yaw_reference_x is not None and not np.isfinite(self.fixed_yaw_reference_x):
            raise ValueError("fixed_yaw_reference_x must be finite.")
        if not self.condition_number_warning > 1.0:
            raise ValueError("condition_number_warning must exceed 1.")

    @property
    def width_mode(self) -> str:
        return "profile" if self.shoe_width_x is not None else "constant"

    def widths_at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.shoe_width_x is None or self.shoe_width_values is None:
            return np.full(x.shape, float(self.shoe_width_m))
        return np.interp(x, np.asarray(self.shoe_width_x, float), np.asarray(self.shoe_width_values, float))

    def metadata(self) -> dict:
        return {
            "out_of_plane_model": OUT_OF_PLANE_MODEL,
            "enabled": bool(self.enabled),
            "shoe_width": float(self.shoe_width_m),
            "shoe_width_profile": None
            if self.shoe_width_x is None
            else {"x_m": list(self.shoe_width_x), "width_m": list(self.shoe_width_values or ())},
            "out_of_plane_shear_model": self.out_of_plane_shear_model,
            "lateral_to_normal_stiffness_ratio": self.lateral_to_normal_stiffness_ratio,
            "yaw_reference": self.yaw_reference,
            "fixed_yaw_reference_x": self.fixed_yaw_reference_x,
            "width_distribution_assumption": WIDTH_DISTRIBUTION_ASSUMPTION,
            "fem_width_convention": FEM_WIDTH_CONVENTION,
            "reference_extrusion_width": "1 m (plane strain)",
            "physical_shoe_width_source": "shoe_width_x/values profile" if self.shoe_width_x is not None else "shoe_width_m",
            "units": dict(OUT_OF_PLANE_UNITS),
        }


@dataclass(frozen=True)
class OutOfPlaneStiffness:
    """Diagonal 2.5D tangent stiffnesses of one converged state (SI)."""

    lateral_translation_stiffness: float  # N/m
    roll_stiffness: float  # N*m/rad
    yaw_stiffness: float  # N*m/rad
    yaw_reference_x: float  # fixed-frame horizontal coordinate of the yaw axis (m)
    yaw_reference_material_x: float  # material x of the yaw axis (m)
    yaw_reference_source: str
    active_contact_node_count: int
    lateral_to_normal_stiffness_ratio: float
    lateral_ratio_source: str
    shoe_width_summary: dict
    status: str
    condition_number: float = float("nan")
    symmetry_error: float = float("nan")
    min_eigenvalue: float = float("nan")
    tangent_kind: str = TANGENT_KIND
    augmented: bool = False
    regularized: bool = False
    near_active_set_change: bool = False
    warnings: tuple[str, ...] = ()
    compute_time_s: float = 0.0
    units: dict = field(default_factory=lambda: dict(OUT_OF_PLANE_UNITS))

    @property
    def diagonal(self) -> np.ndarray:
        """``[k_z, k_roll, k_yaw]`` in the order of ``OUT_OF_PLANE_DOFS``."""
        return np.array([self.lateral_translation_stiffness, self.roll_stiffness, self.yaw_stiffness])

    def as_matrix(self) -> np.ndarray:
        """3x3 tangent in ``OUT_OF_PLANE_DOFS`` order. Coupling terms are not derived yet: NaN off the diagonal."""
        m = np.full((3, 3), np.nan)
        np.fill_diagonal(m, self.diagonal)
        return m


def _nan_result(status: str, *, settings: OutOfPlaneSettings, eta: float, eta_source: str, n_active: int = 0,
                widths: dict | None = None, warnings: tuple[str, ...] = (), zero: bool = False, **kw) -> OutOfPlaneStiffness:
    v = 0.0 if zero else float("nan")
    return OutOfPlaneStiffness(
        lateral_translation_stiffness=v, roll_stiffness=v, yaw_stiffness=v,
        yaw_reference_x=kw.pop("yaw_reference_x", float("nan")),
        yaw_reference_material_x=kw.pop("yaw_reference_material_x", float("nan")),
        yaw_reference_source=kw.pop("yaw_reference_source", settings.yaw_reference),
        active_contact_node_count=int(n_active), lateral_to_normal_stiffness_ratio=float(eta),
        lateral_ratio_source=eta_source, shoe_width_summary=widths or {"mode": settings.width_mode},
        status=status, warnings=warnings, **kw,
    )


def compute_out_of_plane_stiffness(
    op: ContactOperator,
    sol: ContactSolution,
    settings: OutOfPlaneSettings,
    *,
    poisson_ratio: float | None = None,
    compute_eigenvalues: bool = True,
) -> OutOfPlaneStiffness:
    """Diagonal 2.5D tangent stiffnesses about the converged state ``sol`` (read-only).

    ``poisson_ratio`` (contact foam) supplies the default lateral ratio when
    ``settings.lateral_to_normal_stiffness_ratio`` is ``None``. Policy: no
    active node -> zeros (``no_contact``); Cholesky failure -> NaN
    (``tangent_not_positive_definite``); ``1/rcond > 1e14`` -> NaN
    (``tangent_singular``); ``1/rcond > condition_number_warning`` -> values with
    status ``ill_conditioned``. Nothing is regularized; ``sol`` is not modified.
    """
    t0 = time.perf_counter()
    if settings.lateral_to_normal_stiffness_ratio is not None:
        eta, eta_source = float(settings.lateral_to_normal_stiffness_ratio), "user"
    elif poisson_ratio is not None and np.isfinite(poisson_ratio):
        eta, eta_source = lateral_ratio_from_poisson(poisson_ratio), f"assumed thin-layer G/M from nu={poisson_ratio:g}"
    else:
        eta, eta_source = float("nan"), "unavailable"
    if not settings.enabled:
        return _nan_result(STATUS_DISABLED, settings=settings, eta=eta, eta_source=eta_source)

    active = np.flatnonzero(np.asarray(sol.active, dtype=bool))
    x_mat = op.x_bottom[active]
    b = settings.widths_at(x_mat)
    widths = {
        "mode": settings.width_mode,
        "min_m": float(b.min()) if b.size else float("nan"),
        "max_m": float(b.max()) if b.size else float("nan"),
        "mean_m": float(b.mean()) if b.size else float("nan"),
    }
    # Yaw axis: fixed-frame horizontal position of the COP or of the material point (x_ref, 0).
    c, s = float(np.cos(sol.varphi)), float(np.sin(sol.varphi))
    n = op.n_bottom
    xF = sol.T[0] + c * (op.x_bottom + sol.d_b[:n]) - s * (op.y_bottom + sol.d_b[n:])
    x_ref_mat = float(settings.fixed_yaw_reference_x) if settings.fixed_yaw_reference_x is not None else 0.5 * op.L
    lam_tot = float(sol.lam.sum())
    yaw_kw: dict = {}
    if settings.yaw_reference == YAW_REFERENCE_COP and lam_tot > float(sol.tolerances["reaction_tolerance"]):
        yaw_x = float(sol.lam @ xF / lam_tot)
        yaw_kw = {"yaw_reference_x": yaw_x, "yaw_reference_material_x": float(sol.lam @ op.x_bottom / lam_tot),
                  "yaw_reference_source": YAW_REFERENCE_COP}
    else:
        yaw_x = float(sol.T[0] + c * x_ref_mat)
        source = YAW_REFERENCE_SHOE if settings.yaw_reference == YAW_REFERENCE_SHOE else "shoe_reference (no load for COP)"
        yaw_kw = {"yaw_reference_x": yaw_x, "yaw_reference_material_x": x_ref_mat, "yaw_reference_source": source}

    if active.size == 0:
        return _nan_result(STATUS_NO_CONTACT, settings=settings, eta=eta, eta_source=eta_source, widths=widths,
                           zero=True, compute_time_s=time.perf_counter() - t0, **yaw_kw)

    warnings: list[str] = []
    # Pure-normal active compliance (mu_eff = 0): H = N G N^T, symmetric positive definite.
    H = _normal_operator(op, c, s, s, c)[np.ix_(active, active)]
    sym_err = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
    H = 0.5 * (H + H.T)
    try:
        cho = linalg.cho_factor(H, lower=False, check_finite=True)
    except linalg.LinAlgError:
        return _nan_result(STATUS_NOT_PD, settings=settings, eta=eta, eta_source=eta_source, n_active=active.size,
                           widths=widths, symmetry_error=sym_err, compute_time_s=time.perf_counter() - t0, **yaw_kw)
    rcond, _ = linalg.lapack.dpocon(cho[0], float(np.linalg.norm(H, 1)))
    cond = float(1.0 / rcond) if rcond > 0 else float("inf")
    min_eig = float(np.linalg.eigvalsh(H)[0]) if compute_eigenvalues else float("nan")
    if not cond < SINGULAR_CONDITION:
        return _nan_result(STATUS_SINGULAR, settings=settings, eta=eta, eta_source=eta_source, n_active=active.size,
                           widths=widths, condition_number=cond, symmetry_error=sym_err, min_eigenvalue=min_eig,
                           compute_time_s=time.perf_counter() - t0, **yaw_kw)
    status = STATUS_OK
    if cond > settings.condition_number_warning:
        status = STATUS_ILL_CONDITIONED
        warnings.append(f"active normal tangent condition {cond:.3e} exceeds {settings.condition_number_warning:.1e}")

    def quad(d: np.ndarray) -> float:
        return float(d @ linalg.cho_solve(cho, d))

    r = xF[active] - yaw_x
    sb = np.sqrt(b)
    k_normal_translation = quad(sb)
    k_yaw_normal = quad(sb * r)
    k_roll = quad(np.sqrt(b**3 / 12.0))
    if not np.isfinite(eta):
        warnings.append("lateral-to-normal ratio unavailable: k_z and k_yaw are NaN")
        status = STATUS_NO_RATIO
    k_z = eta * k_normal_translation
    k_yaw = eta * k_yaw_normal

    # Close to an active-set change: an inactive gap or an active reaction near zero.
    frac = float(settings.near_change_fraction)
    inactive = ~np.asarray(sol.active, dtype=bool)
    near = bool(
        (inactive.any() and float(np.min(sol.gap[inactive])) <= frac * op.h_mesh)
        or float(np.min(sol.lam[active])) <= frac * float(np.max(sol.lam[active]))
    )
    if near:
        warnings.append("active set is close to changing: the tangent is one-sided and may jump at the next transition")
    if settings.yaw_reference == YAW_REFERENCE_COP and yaw_kw["yaw_reference_source"] != YAW_REFERENCE_COP:
        warnings.append("no normal load: yaw reference fell back to the shoe reference point")
    return OutOfPlaneStiffness(
        lateral_translation_stiffness=float(k_z),
        roll_stiffness=float(k_roll),
        yaw_stiffness=float(k_yaw),
        active_contact_node_count=int(active.size),
        lateral_to_normal_stiffness_ratio=float(eta),
        lateral_ratio_source=eta_source,
        shoe_width_summary=widths,
        status=status,
        condition_number=cond,
        symmetry_error=sym_err,
        min_eigenvalue=min_eig,
        near_active_set_change=near,
        warnings=tuple(warnings),
        compute_time_s=time.perf_counter() - t0,
        **yaw_kw,
    )
