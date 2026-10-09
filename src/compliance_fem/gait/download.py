"""Dataset download helpers and local manifest for Wang et al. (2025)."""

from __future__ import annotations

import json
from pathlib import Path

ARTICLE_DOI = "https://doi.org/10.1038/s41597-025-05113-6"
RECORDS = {
    "c3d": {
        "doi": "https://doi.org/10.17608/k6.auckland.27015301",
        "figshare_article_id": 27015301,
        "note": "Raw C3D motion-capture + forces",
    },
    "trc": {
        "doi": "https://doi.org/10.17608/k6.auckland.27015298",
        "figshare_article_id": 27015298,
        "note": "OpenSim TRC marker trajectories",
    },
    "mot": {
        "doi": "https://doi.org/10.17608/k6.auckland.27015517",
        "figshare_article_id": 27015517,
        "note": "OpenSim MOT ground reactions",
    },
    "csv": {
        "doi": "https://doi.org/10.17608/k6.auckland.27015535",
        "figshare_article_id": 27015535,
        "note": "CSV exports",
    },
}

MANIFEST_TEXT = """# Wang et al. gait dataset (local cache)

Publication: Yuan Wang et al., Scientific Data 12, 802 (2025).
Article: https://doi.org/10.1038/s41597-025-05113-6

Download the matched OpenSim TRC + MOT records from the University of Auckland
Figshare pages (browser download may be required due to bot protection):

- TRC: https://doi.org/10.17608/k6.auckland.27015298
- MOT: https://doi.org/10.17608/k6.auckland.27015517

Place extracted files under this directory, preserving names like
`P01_pr1_01.trc` / `P01_pr1_01.mot` (confirm exact stems on Figshare).

Reported conventions (verify against files):
- markers 200 Hz, forces 1000 Hz
- +x anterior, +y superior, +z right
- markers/COP in mm → converted to m by the adapter
- forces in N; moments often N·mm → converted to N·m
- stance threshold 20 N vertical GRF

Cite both the article DOI and the dataset DOIs when publishing results.
"""


def write_data_readme(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "README.md"
    path.write_text(MANIFEST_TEXT, encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"article_doi": ARTICLE_DOI, "records": RECORDS}, indent=2),
        encoding="utf-8",
    )
    return path
