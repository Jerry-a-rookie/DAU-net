# DAU-Net Public Release Checklist

This checklist records the remaining items to verify before publishing the research code.

## Repository contents

- [x] README with installation, dataset layout, training, evaluation, and output instructions.
- [x] Source code for the model, data loaders, and experiment runner.
- [x] `requirements.txt` with installable dependency constraints.
- [x] `LICENSE` and `NOTICE.md`.
- [x] `.gitignore` for datasets, checkpoints, logs, generated outputs, caches, and local environments.
- [x] Citation template in `README.md`.

## Before publishing

- [ ] Confirm the final author list, repository URL, paper DOI, and contact information.
- [ ] Confirm that the selected software license is compatible with every redistributed component.
- [ ] Confirm that dataset files are not uploaded unless their original licenses permit redistribution.
- [ ] Run the quick-start command on at least one dataset.
- [ ] Run a CPU smoke test and a GPU experiment if GPU results are reported.
- [ ] Verify that no checkpoints, logs, private notes, paper drafts, or machine-specific paths are committed.
- [ ] Record the Python, PyTorch, CUDA, and dataset versions used for the reported results.

## Release archive

Package the repository after generated files have been removed or ignored. Do not include local datasets or checkpoints by default. A public release should contain the source code and documentation needed to reproduce the experiments, while users obtain benchmark data from the original providers.
