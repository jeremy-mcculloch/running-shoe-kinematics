"""Contact-lookup and GUI-shape compatibility for layered compliance files."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance, save_compliance_npz
from compliance_fem.config import LayeredPlateConfig
from compliance_fem.contact_lookup import (
    from_compliance_result,
    generate_contact_lookup,
    load_contact_lookup,
    load_force_compliance,
    save_contact_lookup,
)
from compliance_fem.force_control import evaluate_from_angles
from compliance_fem.shape_render import build_shape_plot_data


@pytest.fixture
def layered_config() -> LayeredPlateConfig:
    return LayeredPlateConfig(
        L=0.30,
        h1_heel=0.025,
        h1_toe=0.015,
        h2_heel=0.020,
        h2_toe=0.030,
        E1=2.0e6,
        nu1=0.30,
        E_heel=5.0e5,
        E_toe=1.5e6,
        nu2=0.30,
        EI_plate=10.0,
        nx=6,
        ny1=2,
        ny2=2,
        element_order=1,
    )


@pytest.fixture
def layered_result(layered_config: LayeredPlateConfig):
    return compute_compliance(layered_config)


def test_lookup_from_layered_compliance(layered_result, tmp_path) -> None:
    npz_path = tmp_path / "compliance_results.npz"
    save_compliance_npz(layered_result, npz_path)
    blocks = load_force_compliance(npz_path)
    assert blocks.geometry_type == layered_result.config.geometry_type
    assert blocks.y_top is not None
    assert blocks.y_plate is not None
    lookup = generate_contact_lookup(blocks, a=0.18, kappa=30.0)
    out = tmp_path / "lookup"
    save_contact_lookup(lookup, out)
    reloaded = load_contact_lookup(out / "contact_lookup.npz")
    assert reloaded.geometry_type == blocks.geometry_type
    assert reloaded.y_top is not None
    assert reloaded.y_plate is not None
    np.testing.assert_allclose(reloaded.y_plate, blocks.y_plate)


def test_from_compliance_result_geometry(layered_result) -> None:
    blocks = from_compliance_result(layered_result)
    assert blocks.geometry_type == "layered_trapezoids_inextensible_plate"
    assert blocks.y_plate is not None
    lookup = generate_contact_lookup(blocks, a=0.18, kappa=30.0)
    assert lookup.y_plate is not None


def test_shape_plot_uses_profiles(layered_result) -> None:
    lookup = generate_contact_lookup(from_compliance_result(layered_result), a=0.18, kappa=30.0)
    sel = evaluate_from_angles(lookup, 0.0, -1.0e3, -3.0, 2.0)
    if sel.selected_row is None:
        pytest.skip("No contact candidate selected for this coarse layered mesh.")
    shape = build_shape_plot_data(lookup, sel, scale_mode="true")
    assert shape.y_plate_ref is not None
    assert shape.plate_is_schematic
    assert shape.x_plate_def is not None and shape.y_plate_def is not None
    assert not np.allclose(shape.y_plate_ref, shape.y_plate_ref[0])
    np.testing.assert_allclose(shape.y_bottom_ref, 0.0, atol=1e-12)
    assert "schematic" in shape.caption
