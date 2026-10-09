"""Measured-sole JSON configuration: parsing, precedence, fingerprint, build and runtime."""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_measured_sole import _subsampled_rows, _write_rows

from compliance_fem.api import SoleModel
from compliance_fem.contact.lookup import load_contact_lookup
from compliance_fem.gui.params import gui_params_from_setup, setup_from_gui
from compliance_fem.contact.model_setup import (
    FINGERPRINT_METADATA_KEY,
    SETUP_METADATA_KEY,
    ConfigFileError,
    ensure_lookup_exists,
    load_model_setup,
    lookup_stores_nodal_fields,
    matching_prebuilt_lookup,
    save_model_setup,
    model_setup_from_dict,
    stored_lookup_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "setup.json"
CSV = ROOT / "data" / "geometry" / "sole_geometry.csv"


def _minimal(csv=CSV, **sections) -> dict:
    sole = {"geometry_csv": str(csv), "shoe_length_mm": 270.0, **sections.pop("sole", {})}
    return {"sole": sole, **sections}


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- parsing --------------------------------------------------------------------------


def test_example_config_loads() -> None:
    s = load_model_setup(EXAMPLE)
    assert Path(s.sole.geometry_csv) == CSV.resolve()
    assert s.sole.shoe_length_mm == 270.0 and s.sole.mesh_size_m == pytest.approx(0.003)
    assert s.toe_length_m == pytest.approx(0.0594) and s.softplus_kappa == 160.0
    assert s.toe_spring.toe_stiffness_Nm_per_rad == 25.0 and s.runtime.shoe_width_m == 0.1
    assert s.output_dir == (ROOT / "outputs" / "measured_sole_lookup").resolve()


def test_minimal_config_uses_defaults() -> None:
    s = model_setup_from_dict(_minimal())
    assert s.sole.upper_foam_material == "FFTurbo" and s.sole.ffleap_E_heel_Pa == 3.54e5
    assert s.lookup.toe_length_mm == pytest.approx(59.4) and s.output_dir is None
    assert s.lookup.min_bend_radius_mm == pytest.approx(6.75)
    assert s.toe_length_m == pytest.approx(0.0594) and s.softplus_kappa == pytest.approx(160.0)
    assert s.runtime.shoe_width_m == 0.10


def test_relative_paths_resolve_against_config_dir(tmp_path) -> None:
    sub = tmp_path / "cfg"
    sub.mkdir()
    shutil.copy(CSV, tmp_path / "sole.csv")
    path = _write(sub / "c.json", _minimal(csv="../sole.csv", lookup={"output_dir": "out"}))
    s = load_model_setup(path)
    assert Path(s.sole.geometry_csv) == (tmp_path / "sole.csv").resolve()
    assert s.output_dir == (sub / "out").resolve()


def test_round_trip(tmp_path) -> None:
    s = load_model_setup(EXAMPLE)
    s2 = load_model_setup(save_model_setup(s, tmp_path / "rt.json"))
    assert s2.to_dict() == s.to_dict()
    assert s2.fingerprint() == s.fingerprint()


@pytest.mark.parametrize(
    "data, match",
    [
        ({**_minimal(), "sol": {}}, "Unknown top-level"),
        (_minimal(sole={"mesh_size_mm": 3.0}), "unknown key"),
        (_minimal(sole={"normalized_geometry": None}), "unknown key"),
        ({"sole": {"geometry_csv": str(CSV)}}, "shoe_length_mm is required"),
        (_minimal(lookup={"toe_length_m": 0.06}), "unknown key"),
        (_minimal(lookup={"toe_length_mm": 0.0}), "toe_length_mm"),
        (_minimal(lookup={"toe_length_mm": 300.0}), "shorter than the shoe"),
        (_minimal(lookup={"min_bend_radius_mm": 0.0}), "min_bend_radius_mm"),
        (_minimal(sole={"mesh_size_m": True}), "must be a number"),
        (_minimal(sole={"mesh_size_m": "3"}), "must be a number"),
        (_minimal(sole={"element_order": 1.5}), "integer"),
        (_minimal(sole={"upper_foam_material": "EVA"}), "foam material"),
        ({**_minimal(), "schema_version": 1}, "schema_version"),
        (_minimal(runtime={"shoe_width_m": 0.0}), "shoe_width_m"),
        (_minimal(sole={"ffturbo_nu": 0.6}), "ffturbo_nu"),
        (_minimal(toe_spring={"toe_model": "nope"}), "toe_model"),
    ],
)
def test_invalid_configs_rejected(data, match) -> None:
    with pytest.raises(ConfigFileError, match=match):
        model_setup_from_dict(data)


def test_invalid_json_and_missing_file(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigFileError, match="invalid JSON"):
        load_model_setup(bad)
    with pytest.raises(FileNotFoundError):
        load_model_setup(tmp_path / "missing.json")


# --- fingerprint ------------------------------------------------------------------------


def test_fingerprint_tracks_build_parameters_only(tmp_path) -> None:
    base = model_setup_from_dict(_minimal())
    fp = base.fingerprint()
    for change in (
        _minimal(sole={"ffturbo_E_Pa": 3.0e5}),
        _minimal(sole={"mesh_size_m": 0.004}),
        _minimal(lookup={"min_bend_radius_mm": 10.0}),
        _minimal(lookup={"toe_length_mm": 70.0}),
        _minimal(sole={"shoe_length_mm": 280.0}),
    ):
        assert model_setup_from_dict(change).fingerprint() != fp
    shutil.copy(CSV, tmp_path / "copy.csv")
    same = model_setup_from_dict(
        _minimal(csv=tmp_path / "copy.csv", toe_spring={"toe_stiffness_Nm_per_rad": 5.0}, runtime={"shoe_width_m": 0.2})
    )
    assert same.fingerprint() == fp


# --- build / reuse / runtime (coarse geometry) ----------------------------------------------


@pytest.fixture(scope="module")
def coarse_config(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cfg")
    _write_rows(d / "coarse.csv", _subsampled_rows())
    return _write(
        d / "coarse.json",
        {
            "sole": {"geometry_csv": "coarse.csv", "shoe_length_mm": 270.0, "mesh_size_m": 0.010},
            "lookup": {"output_dir": "lookup"},
            "toe_spring": {"toe_stiffness_Nm_per_rad": 20.0},
            "runtime": {"shoe_width_m": 0.09},
        },
    )


@pytest.fixture(scope="module")
def built(coarse_config):
    setup = load_model_setup(coarse_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        npz = ensure_lookup_exists(setup, quiet=True, force=True, skip_plots=True)
    return setup, load_contact_lookup(npz), npz


def test_build_records_setup(built) -> None:
    setup, lookup, npz = built
    assert npz.parent == setup.output_dir
    assert (npz.parent / "generation_summary.json").is_file() and not list(npz.parent.glob("*.png"))
    assert lookup.softplus_toe_length == pytest.approx(setup.toe_length_m)
    assert lookup.softplus_kappa == setup.softplus_kappa
    assert stored_lookup_fingerprint(npz) == setup.fingerprint()
    assert load_model_setup(npz.parent / "measured_sole_config.json").fingerprint() == setup.fingerprint()
    reloaded = load_contact_lookup(npz)
    assert reloaded.metadata[FINGERPRINT_METADATA_KEY] == setup.fingerprint()
    assert reloaded.metadata[SETUP_METADATA_KEY]["runtime"]["shoe_width_m"] == 0.09


def test_ensure_reuses_matching_lookup(built) -> None:
    setup, _, npz = built
    mtime = npz.stat().st_mtime_ns
    assert matching_prebuilt_lookup(setup) == npz
    assert ensure_lookup_exists(setup, quiet=True) == npz
    assert npz.stat().st_mtime_ns == mtime
    stale = replace(setup, sole=replace(setup.sole, EI_plate_Nm2_per_m=3.0))
    assert matching_prebuilt_lookup(stale) is None


def test_sole_model_from_config(built, coarse_config, tmp_path) -> None:
    setup, _, _ = built
    m = SoleModel.from_config(coarse_config)
    assert m.shoe_width_m == 0.09 and m.toe_config.toe_stiffness_Nm_per_rad == 20.0
    st = m.step(-20.0, -400.0, 2.0)
    assert st.valid
    np.testing.assert_allclose(st.top_force_N.sum(axis=1), [-20.0, -400.0], rtol=1e-8)

    data = json.loads(coarse_config.read_text())
    data["sole"]["EI_plate_Nm2_per_m"] = 3.0
    data["sole"]["geometry_csv"] = str(Path(setup.sole.geometry_csv))
    data["lookup"]["output_dir"] = str(setup.output_dir)
    stale = _write(tmp_path / "stale.json", data)
    with pytest.raises(ConfigFileError, match="different parameters"):
        SoleModel.from_config(stale)
    assert SoleModel.from_config(stale, require_matching_lookup=False).shoe_width_m == 0.09


def test_from_config_warns_without_fingerprint(built, coarse_config) -> None:
    _, lookup, _ = built
    lookup.metadata.pop(FINGERPRINT_METADATA_KEY)
    try:
        with pytest.warns(UserWarning, match="fingerprint"):
            SoleModel.from_config(coarse_config, lookup=lookup)
    finally:
        lookup.metadata[FINGERPRINT_METADATA_KEY] = built[0].fingerprint()


def test_lookup_cli_with_config(coarse_config, tmp_path) -> None:
    from compliance_fem.cli.contact_lookup import main

    out = tmp_path / "cli_lookup"
    data = json.loads(coarse_config.read_text())
    data["sole"]["geometry_csv"] = str(coarse_config.parent / data["sole"]["geometry_csv"])
    data["lookup"] = {"output_dir": str(out), "min_bend_radius_mm": 9.0, "store_nodal_fields": True}
    cfg = _write(tmp_path / "cli.json", data)
    with pytest.raises(SystemExit):
        main([str(cfg), "--kappa", "120"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        main([str(cfg), "--quiet", "--save-fem"])
    setup = load_model_setup(out / "measured_sole_config.json")
    assert setup.softplus_kappa == 120.0 and setup.lookup.store_nodal_fields
    assert stored_lookup_fingerprint(out) == setup.fingerprint()
    summary = json.loads((out / "generation_summary.json").read_text())
    assert summary["softplus_kappa"] == 120.0 and summary["model_setup_fingerprint"] == setup.fingerprint()
    assert summary["nodal_fields_stored"] is True and lookup_stores_nodal_fields(out)
    assert load_contact_lookup(out).nodal_fields_stored
    for name in ("compliance_results.npz", "mesh.msh", "mesh.vtk", "Cbb_force_heatmap.png", "fem_summary.json"):
        assert (out / "fem" / name).is_file(), name
    assert list(out.glob("*_interval_heatmap.png"))

    mtime = (out / "contact_lookup.npz").stat().st_mtime_ns
    main([str(cfg), "--quiet"])
    assert (out / "contact_lookup.npz").stat().st_mtime_ns == mtime


# --- nodal field storage ---------------------------------------------------------------------


def test_store_nodal_fields_parsing() -> None:
    assert model_setup_from_dict(_minimal()).lookup.store_nodal_fields is False
    assert model_setup_from_dict(_minimal(lookup={"store_nodal_fields": True})).lookup.store_nodal_fields is True
    with pytest.raises(ConfigFileError, match="store_nodal_fields"):
        model_setup_from_dict(_minimal(lookup={"store_nodal_fields": "yes"}))
    s = model_setup_from_dict(_minimal())
    assert replace(s, lookup=replace(s.lookup, store_nodal_fields=True)).fingerprint() == s.fingerprint()


def test_built_lookup_is_compact_by_default(built) -> None:
    setup, lookup, npz = built
    assert not setup.lookup.store_nodal_fields
    assert not lookup.nodal_fields_stored and lookup.has_nodal_fields
    assert not lookup_stores_nodal_fields(npz)
    with np.load(npz, allow_pickle=True) as data:
        assert not any(k.endswith("_basis") and k.startswith(("bottom_", "reaction_", "top_force_")) for k in data.files)
        assert "field_solver_compliance" in data.files
    wants_fields = replace(setup, lookup=replace(setup.lookup, store_nodal_fields=True))
    assert matching_prebuilt_lookup(wants_fields) is None
    assert matching_prebuilt_lookup(setup) == npz


# --- GUI consistency -------------------------------------------------------------------------


def test_gui_params_round_trip_fingerprint(built) -> None:
    setup, _, npz = built
    assert setup_from_gui(gui_params_from_setup(setup)).fingerprint() == setup.fingerprint()
    from compliance_fem.gui.app import _draft_from_setup

    draft = _draft_from_setup(setup)
    assert draft["prebuilt_lookup"] == str(npz)
    assert setup_from_gui(draft).fingerprint() == setup.fingerprint()
