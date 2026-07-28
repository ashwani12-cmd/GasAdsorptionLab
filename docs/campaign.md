# Campaign tutorial

Use `AdsorptionCampaign` to generate all site/orientation combinations and
their QE job directories.

```python
from gal import AdsorptionCampaign

campaign = AdsorptionCampaign("MoS2.cif", ["CO", "NH3", "NO2"])
jobs = campaign.generate()
```

The campaign root contains `campaign.json`, `campaign.csv`, and a hierarchy
of surface, adsorbate, and site folders. Re-running skips existing jobs;
pass `overwrite=True` to regenerate them. Status is inferred from `pw.out`.

YAML campaigns can be started with:

```bash
gal campaign config.yaml
```
