# Publishing DiffHydro to PyPI

1. **Update metadata**
   * Bump the version in `pyproject.toml`.
   * Ensure `README.md` and `CHANGELOG` entries (if any) are up to date.
2. **Clean previous build artifacts**
   ```bash
   rm -rf build dist diffhydro.egg-info
   ```
3. **Install packaging utilities**
   ```bash
   python -m pip install --upgrade build twine
   ```
4. **Build the distribution**
   ```bash
   python -m build
   ```
   This generates both `dist/diffhydro-<version>.tar.gz` (sdist) and `.whl` (wheel).
5. **Verify the metadata**
   ```bash
   python -m twine check dist/*
   ```
6. **Upload to PyPI**
   ```bash
   python -m twine upload dist/*
   ```
   Use `--repository testpypi` if you want to test the process first.
7. **Tag the release**
   ```bash
   git tag -a v0.1 -m "DiffHydro v0.1"
   git push --tags
   ```
8. **Validate installation**
   ```bash
   python -m pip install --upgrade diffhydro
   ```

Once the package is on PyPI, users can install it via `pip install diffhydro`.
