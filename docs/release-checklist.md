# v1.0 release checklist

- [ ] Set the final semantic version in `pyproject.toml` and `gal.__version__`.
- [ ] Update `CHANGELOG.md` and `CITATION.cff`.
- [ ] Run `python -m pip install -e . --no-deps`.
- [ ] Run `PYTHONPATH=. pytest -q`.
- [ ] Run every script in `examples/`.
- [ ] Build with `python -m build` and inspect `dist/`.
- [ ] Confirm the GitHub Actions matrix is green on Python 3.10–3.12.
- [ ] Tag the release and publish the built sdist and wheel.
