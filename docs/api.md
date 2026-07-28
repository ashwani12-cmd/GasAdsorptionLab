# API reference

## Public package API

| Symbol | Purpose |
| --- | --- |
| `Surface` | ASE-backed surface container; loads CIF files and builds cells. |
| `SiteFinder` | Geometry-driven adsorption-site detector. |
| `Site`, `SiteType`, `SurfaceType` | Site data model and classification enums. |
| `Gas` | Legacy XYZ-backed gas library. |
| `Adsorbate` | ASE-backed adsorbate library and transformations. |
| `place_adsorbate`, `generate_adsorption_structures` | Canonical placement helpers. |
| `AdsorptionWorkflow` | Stepwise adsorption/QE workflow. |
| `AdsorptionCampaign` | Multi-surface campaign and job orchestration. |
| `Config` | YAML/dictionary configuration access. |

## QE API

Import QE utilities from `gal.qe`:

| Symbol | Purpose |
| --- | --- |
| `QEInput` | Low-level pw.x serializer. |
| `QEInputBuilder` | Settings-based QE input builder. |
| `QEJobWriter` | QE job directory writer. |

Internal graph engines, placement helpers beginning with `_`, and scheduler
template helpers are intentionally not part of the supported public API.
