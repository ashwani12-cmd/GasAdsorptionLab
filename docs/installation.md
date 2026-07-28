# Installation

GasAdsorptionLab requires Python 3.10 or newer.

```bash
python -m pip install gasadsorptionlab
```

For development from a checkout:

```bash
python -m pip install -e .
python -m pytest -q
```

The package includes ASE and symmetry dependencies. Materials Project support
is used only when calling `Surface.from_mp`.
