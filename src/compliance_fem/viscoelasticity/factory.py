"""Material configuration dataclasses and factory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Union

from compliance_fem.viscoelasticity.base import ViscoelasticForceMapper
from compliance_fem.viscoelasticity.fractional import FractionalMaterial
from compliance_fem.viscoelasticity.fung import FungMaterial
from compliance_fem.viscoelasticity.sls import SLSMaterial


@dataclass(frozen=True)
class SLSConfig:
    """SLS parameters.

    Units
    -----
    ``g_inf``
        Dimensionless long-term modulus ratio, ``0 < g_inf <= 1``.
    ``tau_r``
        Relaxation time (same units as sample times).
    """

    g_inf: float
    tau_r: float
    n_components: int = 2
    model: Literal["sls"] = "sls"


@dataclass(frozen=True)
class FractionalConfig:
    """Fractional power-law creep parameters.

    Units
    -----
    ``E0``
        Reference modulus (force / area). No unit conversion is performed.
    ``T``
        Characteristic time (same units as sample times).
    ``alpha``
        Dimensionless fractional order, ``0 < alpha < 1``.
    ``tau_min``, ``tau_max``
        Approximation window for the sum-of-exponentials kernel (time units).
        Defaults to ``[1e-6 T, 1e6 T]`` when omitted.
    """

    E0: float
    T: float
    alpha: float
    num_modes: int = 32
    tau_min: float | None = None
    tau_max: float | None = None
    n_components: int = 2
    model: Literal["fractional"] = "fractional"


@dataclass(frozen=True)
class FungConfig:
    """Fung continuous-spectrum parameters.

    Units
    -----
    ``C``
        Dimensionless spectral intensity.
    ``tau1``, ``tau2``
        Spectrum bounds (time units), ``0 < tau1 < tau2``.
    """

    C: float
    tau1: float
    tau2: float
    num_modes: int = 32
    n_components: int = 2
    model: Literal["fung"] = "fung"


MaterialConfig = Union[SLSConfig, FractionalConfig, FungConfig]


def create_material(config: MaterialConfig | dict[str, Any]) -> ViscoelasticForceMapper:
    """Construct a stateful mapper from a config object or plain dict."""
    if isinstance(config, dict):
        model = str(config.get("model", "")).lower()
        data = {k: v for k, v in config.items() if k != "model"}
        if model == "sls":
            config = SLSConfig(**data)
        elif model == "fractional":
            config = FractionalConfig(**data)
        elif model == "fung":
            config = FungConfig(**data)
        else:
            raise ValueError(f"unknown model {model!r}; expected sls|fractional|fung")

    if isinstance(config, SLSConfig):
        return SLSMaterial(
            g_inf=config.g_inf,
            tau_r=config.tau_r,
            n_components=config.n_components,
        )
    if isinstance(config, FractionalConfig):
        return FractionalMaterial(
            E0=config.E0,
            T=config.T,
            alpha=config.alpha,
            num_modes=config.num_modes,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            n_components=config.n_components,
        )
    if isinstance(config, FungConfig):
        return FungMaterial(
            C=config.C,
            tau1=config.tau1,
            tau2=config.tau2,
            num_modes=config.num_modes,
            n_components=config.n_components,
        )
    raise TypeError(f"unsupported config type {type(config)!r}")


def config_to_dict(config: MaterialConfig) -> dict[str, Any]:
    return asdict(config)
