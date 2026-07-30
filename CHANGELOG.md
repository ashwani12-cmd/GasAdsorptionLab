# Changelog

All notable changes to GasAdsorptionLab are documented here.

## Unreleased

### Added

- `QEInputBuilder` now defaults to Grimme-D3 van der Waals dispersion
  (`vdw_corr='grimme-d3'`), Marzari-Vanderbilt smearing (`smearing='mv'`,
  `degauss=0.01`), and `nspin=1`, so generated `pw.in` decks are physically
  correct by default for cases like physisorbed CO on WSe2. Pass
  `vdw_corr=None` to disable dispersion correction.

### Fixed

- `export_optimized_structures()` no longer raises `IndexError` for numeric
  `unique_site_id` values, and no longer overwrites structures when multiple
  adsorbates share a site: parsed structures are now keyed by job directory
  instead of site id.
- `compute_adsorption_energy()` now warns when `clean_surface_energy`/
  `gas_phase_energy` look like they are still in Ry instead of eV. Added
  `CampaignResults.reference_energy()` to parse reference energies in eV
  directly from a pw.out file.
- A relaxation killed before completion (e.g. at walltime) is no longer
  marked `Converged`; `converged` now requires `JOB DONE.` plus the
  appropriate final convergence marker (`bfgs converged` for relaxations,
  `convergence has been achieved` for SCF-only runs). `rank_by_adsorption_energy()`
  and `plot_adsorption_energy()` now exclude non-converged rows by default
  (`only_converged=True`).

## 0.6.0

- Added campaign orchestration, CSV/JSON summaries, restart handling, status
  tracking, and the `gal campaign` command-line interface.
- Added completed Quantum ESPRESSO campaign analysis, adsorption-energy
  ranking, tabular exports, optimized-structure export, and `gal analyze`.

## 0.5.0

- Added ASE-backed adsorbates, automatic placement/orientations, and Quantum
  ESPRESSO input/job-directory generation.

## 0.4.0

- Added public API exports and universal geometry-driven site engines.

## 0.3.0

- Added periodic-image graph handling for primitive adsorption surfaces.

## 0.2.0

- Added adsorption-site, symmetry, visualization, and QE input utilities.

## 0.1.0

- Initial release with surface construction, gas loading, and ASE integration.
