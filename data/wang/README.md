# Wang et al. gait dataset (local cache)

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
