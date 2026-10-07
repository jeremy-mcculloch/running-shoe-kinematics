"""Measured-sole JSON configuration: parsing, precedence, fingerprint, build and runtime."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_measured_sole import _subsampled_rows, _write_rows

from compliance_fem.api import SoleModel
from compliance_fem.cli import add_measured_arguments, measured_setup_from_args
from compliance_fem.contact_lookup import load_contact_lookup
from compliance_fem.gui_params import gui_params_from_setup, measured_setup_from_gui
from compliance_fem.measured_config_file import (
    FINGERPRINT_METADATA_KEY,
    SETUP_METADATA_KEY,
    ConfigFileError,
    build_measured_lookup,
    ensure_measured_lookup,
    load_measured_sole_config,
    lookup_stores_nodal_fields,
    matching_prebuilt_lookup,
    save_measured_sole_config,
    setup_from_dict,
    stored_lookup_fingerprint,
    with_lookup_overrides,
    with_sole_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "measured_sole_270mm.json"
CSV = ROOT / "data" / "geometry" / "sole_geometry_normalized.csv"


def _minimal(csv=CSV, **sections) -> dict:
    return {"geometry": {"csv": str(csv), "shoe_length_mm": 270.0}, **sections}


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- parsing --------------------------------------------------------------------------


def test_example_config_loads() -> None:
    s = load_measured_sole_config(EXAMPLE)
    assert Path(s.sole.geometry_csv) == CSV.resolve()
    assert s.sole.shoe_length_mm == 270.0 and s.sole.mesh_size == pytest.approx(0.003)
    assert s.a == pytest.approx(0.78 * 0.27) and s.kappa == 160.0
    assert s.toe_spring.toe_stiffness_Nm_per_rad == 25.0 and s.runtime.shoe_width_m == 0.1
    assert s.output_dir == (ROOT / "outputs" / "measured_sole_lookup").resolve()


def test_minimal_config_uses_defaults() -> None:
    s = setup_from_dict(_minimal())
    assert s.sole.upper_foam_material == "FFTurbo" and s.sole.ffleap_E_heel == 3.54e5
    assert s.lookup.a_over_length == 0.78 and s.output_dir is None
    assert s.runtime.shoe_width_m == 0.10


def test_relative_paths_resolve_against_config_dir(tmp_path) -> None:
    sub = tmp_path / "cfg"
    sub.mkdir()
    shutil.copy(CSV, tmp_path / "sole.csv")
    path = _write(sub / "c.json", _minimal(csv="../sole.csv", lookup={"output_dir": "out"}))
    s = load_measured_sole_config(path)
    assert Path(s.sole.geometry_csv) == (tmp_path / "sole.csv").resolve()
    assert s.output_dir == (sub / "out").resolve()


def test_round_trip(tmp_path) -> None:
    s = load_measured_sole_config(EXAMPLE)
    s2 = load_measured_sole_config(save_measured_sole_config(s, tmp_path / "rt.json"))
    assert s2.to_dict() == s.to_dict()
    assert s2.fingerprint() == s.fingerprint()


@pytest.mark.parametrize(
    "data, match",
    [
        ({**_minimal(), "geomtry": {}}, "Unknown top-level"),
        (_minimal(mesh={"size_m": 3.0}), "unknown key"),
        (_minimal(materials={"FFTurbo": {"E": 1.0}}), "unknown key"),
        ({"geometry": {"csv": str(CSV)}}, "shoe_length_mm is required"),
        (_minimal(lookup={"a_m": 0.2, "a_over_length": 0.7}), "exactly one"),
        (_minimal(lookup={"a_m": 0.5}), "inside"),
        (_minimal(mesh={"size_mm": True}), "must be a number"),
        (_minimal(mesh={"size_mm": "3"}), "must be a number"),
        (_minimal(mesh={"element_order": 1.5}), "integer"),
        (_minimal(materials={"upper_foam": "EVA"}), "must be one of"),
        ({**_minimal(), "schema_version": 2}, "schema_version"),
        (_minimal(runtime={"shoe_width_m": 0.0}), "shoe_width_m"),
        (_minimal(materials={"FFTurbo": {"nu": 0.6}}), "ffturbo_nu"),
        (_minimal(toe_spring={"model": "nope"}), "toe_model"),
    ],
)
def test_invalid_configs_rejected(data, match) -> None:
    with pytest.raises(ConfigFileError, match=match):
        setup_from_dict(data)


def test_invalid_json_and_missing_file(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigFileError, match="invalid JSON"):
        load_measured_sole_config(bad)
    with pytest.raises(FileNotFoundError):
        load_measured_sole_config(tmp_path / "missing.json")


# --- fingerprint ------------------------------------------------------------------------


def test_fingerprint_tracks_build_parameters_only(tmp_path) -> None:
    base = setup_from_dict(_minimal())
    fp = base.fingerprint()
    for change in (
        _minimal(materials={"FFTurbo": {"E_Pa": 3.0e5}}),
        _minimal(mesh={"size_mm": 4.0}),
        _minimal(lookup={"kappa": 100.0}),
        _minimal(lookup={"a_over_length": 0.7}),
        {"geometry": {"csv": str(CSV), "shoe_length_mm": 280.0}},
    ):
        assert setup_from_dict(change).fingerprint() != fp
    shutil.copy(CSV, tmp_path / "copy.csv")
    same = setup_from_dict(
        _minimal(csv=tmp_path / "copy.csv", toe_spring={"stiffness_Nm_per_rad": 5.0}, runtime={"shoe_width_m": 0.2})
    )
    assert same.fingerprint() == fp
    assert setup_from_dict(_minimal(lookup={"a_m": 0.78 * 0.27})).fingerprint() == fp


# --- CLI precedence -----------------------------------------------------------------------


def _parse(argv):
    p = argparse.ArgumentParser()
    add_measured_arguments(p)
    return p.parse_args(argv)


def test_cli_flags_override_config() -> None:
    s = measured_setup_from_args(_parse(["--config", str(EXAMPLE), "--mesh-size-mm", "5", "--upper-foam", "FFLeap"]))
    assert s.sole.mesh_size == pytest.approx(0.005) and s.sole.upper_foam_material == "FFLeap"
    assert s.sole.EI_plate == 2.0 and s.kappa == 160.0
    plain = measured_setup_from_args(_parse(["--config", str(EXAMPLE)]))
    assert plain.fingerprint() == load_measured_sole_config(EXAMPLE).fingerprint()


def test_cli_without_config_requires_length() -> None:
    with pytest.raises(SystemExit):
        measured_setup_from_args(_parse([]))
    s = measured_setup_from_args(_parse(["--shoe-length-mm", "250"]))
    assert s.sole.shoe_length_mm == 250.0 and s.a == pytest.approx(0.78 * 0.25)


# --- build / reuse / runtime (coarse geometry) ----------------------------------------------


@pytest.fixture(scope="module")
def coarse_config(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cfg")
    _write_rows(d / "coarse.csv", _subsampled_rows())
    return _write(
        d / "coarse.json",
        {
            "geometry": {"csv": "coarse.csv", "shoe_length_mm": 270.0},
            "mesh": {"size_mm": 10.0},
            "lookup": {"output_dir": "lookup"},
            "toe_spring": {"stiffness_Nm_per_rad": 20.0},
            "runtime": {"shoe_width_m": 0.09},
        },
    )


@pytest.fixture(scope="module")
def built(coarse_config):
    setup = load_measured_sole_config(coarse_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, lookup, npz = build_measured_lookup(setup)
    return setup, lookup, npz


def test_build_records_setup(built) -> None:
    setup, lookup, npz = built
    assert npz.parent == setup.output_dir
    assert lookup.softplus_a == pytest.approx(setup.a) and lookup.softplus_kappa == setup.kappa
    assert stored_lookup_fingerprint(npz) == setup.fingerprint()
    assert load_measured_sole_config(npz.parent / "measured_sole_config.json").fingerprint() == setup.fingerprint()
    reloaded = load_contact_lookup(npz)
    assert reloaded.metadata[FINGERPRINT_METADATA_KEY] == setup.fingerprint()
    assert reloaded.metadata[SETUP_METADATA_KEY]["runtime"]["shoe_width_m"] == 0.09


def test_ensure_reuses_matching_lookup(built) -> None:
    setup, _, npz = built
    mtime = npz.stat().st_mtime_ns
    assert matching_prebuilt_lookup(setup) == npz
    assert ensure_measured_lookup(setup) == npz
    assert npz.stat().st_mtime_ns == mtime
    stale = with_sole_overrides(setup, EI_plate=3.0)
    assert matching_prebuilt_lookup(stale) is None


def test_sole_model_from_config(built, coarse_config, tmp_path) -> None:
    setup, _, _ = built
    m = SoleModel.from_config(coarse_config)
    assert m.shoe_width_m == 0.09 and m.toe_config.toe_stiffness_Nm_per_rad == 20.0
    st = m.step(-20.0, -400.0, 2.0)
    assert st.valid
    np.testing.assert_allclose(st.top_force_N.sum(axis=1), [-20.0, -400.0], rtol=1e-8)

    data = json.loads(coarse_config.read_text())
    data["materials"] = {"plate": {"EI_Nm2_per_m": 3.0}}
    data["geometry"]["csv"] = str(Path(setup.sole.geometry_csv))
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
    from compliance_fem.contact_lookup_cli import main

    out = tmp_path / "cli_lookup"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        main(["--config", str(coarse_config), "--output", str(out), "--kappa", "120", "--quiet", "--store-fields"])
    setup = load_measured_sole_config(out / "measured_sole_config.json")
    assert setup.kappa == 120.0 and setup.lookup.store_nodal_fields
    assert stored_lookup_fingerprint(out) == setup.fingerprint()
    summary = json.loads((out / "generation_summary.json").read_text())
    assert summary["softplus_kappa"] == 120.0 and summary["measured_setup_fingerprint"] == setup.fingerprint()
    assert summary["nodal_fields_stored"] is True and lookup_stores_nodal_fields(out)
    assert load_contact_lookup(out).nodal_fields_stored


# --- nodal field storage ---------------------------------------------------------------------


def test_store_nodal_fields_parsing() -> None:
    assert setup_from_dict(_minimal()).lookup.store_nodal_fields is False
    assert setup_from_dict(_minimal(lookup={"store_nodal_fields": True})).lookup.store_nodal_fields is True
    with pytest.raises(ConfigFileError, match="store_nodal_fields"):
        setup_from_dict(_minimal(lookup={"store_nodal_fields": "yes"}))
    s = setup_from_dict(_minimal())
    assert with_lookup_overrides(s, store_nodal_fields=True).fingerprint() == s.fingerprint()


def test_built_lookup_is_compact_by_default(built) -> None:
    setup, lookup, npz = built
    assert not setup.lookup.store_nodal_fields
    assert not lookup.nodal_fields_stored and lookup.has_nodal_fields
    assert not lookup_stores_nodal_fields(npz)
    with np.load(npz, allow_pickle=True) as data:
        assert not any(k.endswith("_basis") and k.startswith(("bottom_", "reaction_", "top_force_")) for k in data.files)
        assert "field_solver_compliance" in data.files
    wants_fields = with_lookup_overrides(setup, store_nodal_fields=True)
    assert matching_prebuilt_lookup(wants_fields) is None
    assert matching_prebuilt_lookup(setup) == npz


# --- GUI consistency -------------------------------------------------------------------------


def test_gui_params_round_trip_fingerprint(built) -> None:
    setup, _, npz = built
    assert measured_setup_from_gui(gui_params_from_setup(setup)).fingerprint() == setup.fingerprint()
    from compliance_fem.app import _measured_draft_from_setup

    layered = {"h1_heel": 0.02, "h1_toe": 0.01, "h2_heel": 0.02, "h2_toe": 0.03, "nx": 10, "ny1": 2, "ny2": 2, "element_order": 1}
    draft = _measured_draft_from_setup(setup, layered)
    assert draft["prebuilt_lookup"] == str(npz)
    assert measured_setup_from_gui(draft).fingerprint() == setup.fingerprint()
