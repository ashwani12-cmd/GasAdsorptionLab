# Quantum ESPRESSO tutorial

```python
from gal.qe import QEInputBuilder, QEJobWriter

builder = QEInputBuilder(pseudo_dir="./pseudo", ecutwfc=60, ecutrho=480, kpts=(6, 6, 1))
writer = QEJobWriter(builder, "jobs", slurm={"cores": 16, "walltime": "02:00:00"})
job = writer.write_job(structure, "Top_CO", {"surface": "MoS2", "adsorbate": "CO", "site": "Top"})
```

Each directory contains `pw.in`, `submit.sh`, and `metadata.json`.
Pseudopotential filenames can be overridden through `QEInputBuilder`.
