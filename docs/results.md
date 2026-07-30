# Results analysis

`CampaignResults` discovers job folders containing `metadata.json` and parses
their Quantum ESPRESSO `pw.out` files. Incomplete jobs remain in the dataframe
with unavailable values instead of causing analysis to fail.

```python
from gal import CampaignResults

results = CampaignResults.from_directory("campaigns/WSe2_CO")
results.compute_adsorption_energy(clean_surface_energy=-1000.0, gas_phase_energy=-10.0)
print(results.rank_by_adsorption_energy())
results.to_csv()
results.to_excel()
results.to_json()
results.plot_adsorption_energy()
results.export_optimized_structures()
```

QE energies are normalized to eV. The parser registry is designed so parsers
for other electronic-structure codes can be added without changing the public
`CampaignResults` API.

The command-line equivalent is:

```bash
gal analyze campaigns/WSe2_CO --clean-surface-energy -1000.0 --gas-phase-energy -10.0
```
