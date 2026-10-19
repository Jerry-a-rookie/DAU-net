# DAU-Net

Official implementation of **DAU-Net: A Scale-Adaptive U-Net for Multivariate Time Series Forecasting**.


## Installation

Python 3.9--3.12 is recommended. Create an isolated environment and install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code automatically uses CUDA when it is available and `--use_gpu True` is specified. Use `--use_gpu False` for CPU execution.

## Dataset preparation

Dataset files are intentionally not included in this repository. Download them from their original public sources and place them under `./dataset/` using the layouts below.

| Dataset option | Expected path | Expected format |
|---|---|---|
| `ETTh1`, `ETTh2` | `dataset/ETT-small/ETTh1.csv` or `ETTh2.csv` | CSV with a `date` column followed by numeric variables |
| `ETTm1`, `ETTm2` | `dataset/ETT-small/ETTm1.csv` or `ETTm2.csv` | CSV with a `date` column followed by numeric variables |
| `custom` | User-specified `--root_path` and `--data_path` | CSV with a `date` column and numeric variables |
| `illness` | User-specified `--root_path` and `--data_path` | CSV in the same format as the custom loader |
| `PEMS03`--`PEMS08` | User-specified `--root_path` and `--data_path` | `.npz`; the first array has shape `[time, node, channel]` and channel 0 is used |
| `solar` | User-specified `--root_path` and `--data_path` | CSV/TXT with a timestamp column followed by plant readings |


## License and citation

The source code is released under the MIT License. Dataset files are not covered by this repository's license; users must follow the terms of the original dataset providers.

If you use this code, please cite:

```bibtex
@article{daunet2026,
  title   = {DAU-Net: A Scale-Adaptive U-Net for Multivariate Time Series Forecasting},
  journal = {Information Sciences},
  year    = {2026}
}
```

Before publishing the repository, replace or extend the citation metadata with the final author list, DOI, and publication information when available.

## Contact

Please use the repository issue tracker for bug reports, reproducibility questions, and suggestions for improvements.
