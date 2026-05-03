# Madrid Air Quality Analytics Pipeline

**Advanced Coding for Data Analytics | Luiss Guido Carli University**

Beren Cay | Defne Turcan

A complete analytics pipeline for the [METRAQ Air Quality Dataset](https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality) (Madrid, 2001-2024).

---

## Requirements

Python 3.10+ is required.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

`pyarrow` is optional but recommended. It speeds up the Task 8 partitioning step significantly:

```bash
pip install pyarrow
```

---

## Dataset

Download the full dataset from Hugging Face:

```
https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality
```

The pipeline expects a folder containing the yearly CSV files named:

```
metraq_aq-2001.csv
metraq_aq-2002.csv
...
metraq_aq-2024.csv
```

A sample CSV (approximately 100,000 rows) is also supported for quick testing. Pass the file path directly instead of the folder.

---

## How to Run

```bash
python madrid.py /path/to/METRAQ-Air-Quality/
```

If `python` is not found (e.g. on Windows with a virtual environment), use:

```bash
.\.venv\Scripts\python.exe .\madrid.py ".\METRAQ-Air-Quality"
```

To run on the sample file:

```bash
python madrid.py sample.csv
```

The script runs all 10 tasks sequentially in a single execution. No flags or subcommands are needed.

---

## Output

All outputs are written automatically to subdirectories created in the working directory:

| Directory | Contents |
|---|---|
| `figures/` | All plots and charts (PNG, 150 dpi) |
| `corr_matrices/` | Per-year, per-sensor correlation matrices (Task 8) |
| `partitions/` | Yearly data partitions used by parallel workers (Task 8) |

The `figures/` folder also contains `task3_best_imputation_methods.csv`, a summary of the best imputation method selected per pollutant.

---

## Configuration

Key constants at the top of `madrid.py` can be adjusted before running:

| Constant | Default | Description |
|---|---|---|
| `SELECTED_POLLUTANTS` | `["NO2", "O3", "<PM10", "SO2"]` | Pollutants used across Tasks 3-10 |
| `KNN_K` | `3` | Number of neighbours for spatial KNN imputation |
| `CORR_THR` | `0.60` | Correlation threshold for network edges (Tasks 6, 8) |
| `TASK8_MAX_WORKERS` | `2` | Parallel worker count. Set to your physical core count |
| `TASK8_CORR_MIN_PERIODS` | `168` | Minimum overlapping hours required before computing a correlation |
| `TASK3_EVAL_MASK_FRAC` | `0.03` | Fraction of values masked for imputation evaluation |
| `USE_PARTITIONS` | `True` | Pre-split data into yearly partitions before Task 8 |

---

## What Each Task Does

| Task | Description |
|---|---|
| 1 | Load dataset, inspect schema, compute descriptive statistics, produce distribution plots |
| 2 | Reconstruct original missingness from `is_interpolated`, detect invalid values, analyse temporal and sensor-specific gaps |
| 3 | Compare imputation methods (spatial KNN, rolling mean, ffill/bfill), evaluate via pseudo-gap RMSE, select best validation-based method per pollutant |
| 4 | Temporal analysis: yearly trends, monthly seasonality, diurnal profiles, year x month heatmaps |
| 5 | Build spatial network (KNN k=2) from UTM coordinates, compute graph metrics, detect communities |
| 6 | Build correlation network from sensor time-series similarity, compare against spatial network |
| 7 *(optional)* | Graph Laplacian diffusion model for NO2 propagation across the sensor network |
| 8 | Parallelised per-year, per-sensor correlation matrices; sequential vs parallel runtime comparison |
| 9 *(optional)* | 24-hour-ahead NO2 forecasting with Ridge regression and Random Forest using meteorological and traffic features |
| 10 | Final summary panels combining key results from all tasks |

---

## Main Results

- NO2 shows a strong long-term decrease from 2001 to 2024, while seasonal and daily cycles remain clearly visible across all stations.
- Spatial KNN imputation performs best for NO2, O3, and PM10 under pseudo-gap evaluation. SO2 is better handled by a 24-hour past rolling mean due to its more localised emission sources.
- The spatial network is built using k-nearest neighbours because simple distance thresholds produce disconnected components or fully connected graphs depending on the chosen value.
- The NO2 correlation network is denser than the spatial graph, suggesting that city-wide temporal drivers (meteorology, traffic rhythms) dominate over geographic proximity for this pollutant.
- Task 8 parallelisation reduces runtime while producing identical correlation outputs, verified through matrix hashing.
- The 24-hour NO2 forecasting model outperforms the persistence baseline, with Random Forest achieving the lowest test error and recent NO2 concentration as the dominant predictor.

---

## Expected Runtime (full dataset, 2 CPU cores, SSD)

| Task | Approx. time |
|---|---|
| 1: Load + Inspect | 5-10 min |
| 2: Missingness | 15-20 min |
| 3: Imputation | 25-40 min |
| 4: Temporal | 5-10 min |
| 5: Spatial network | less than 1 min |
| 6: Correlation network | 3-5 min |
| 7: Propagation model | 10-15 min |
| 8: Parallelisation | approximately 15 min |
| 9: Forecasting | 5-8 min |
| 10: Final plots | less than 1 min |
| **Total** | **approximately 60-90 min** |

Peak RAM usage during optimised loading is approximately 2 GB, but some tasks may require more depending on the machine, pandas version, and intermediate copies created during processing.

---

## Repository Structure

```
.
├── madrid.py          # Main pipeline script
├── requirements.txt   # Python dependencies
├── README.md
├── sample.csv         # Small sample for testing
├── figures/           # Created on first run
├── corr_matrices/     # Created by Task 8
└── partitions/        # Created by Task 8
```

---

## Known Issues, Error Categories, and Practical Guidelines

### 1. Dataset path errors

If the script raises `FileNotFoundError`, check that the path points either to the folder containing the yearly files (`metraq_aq-2001.csv` ... `metraq_aq-2024.csv`) or to a single CSV file such as the sample dataset.

```bash
python madrid.py /path/to/METRAQ-Air-Quality/   # directory mode
python madrid.py sample.csv                      # single-file mode

# On Windows with a virtual environment:
.\.venv\Scripts\python.exe .\madrid.py ".\METRAQ-Air-Quality"
```

### 2. Memory limitations

The full dataset has approximately 64M rows. The script uses memory-efficient dtypes and year-partitioned processing, but on low-memory machines it is advisable to first test the pipeline on `sample.csv` before running the full dataset.

### 3. Missingness is not represented by NaN

The METRAQ dataset already filled all missing values before release. There are no NaN values in the raw data. Original missingness is reconstructed using the flag:

```python
is_interpolated == True
```

Imputation comparisons are therefore based on this flag, not on NaN detection.

### 4. Variable availability differs by period

Not all variables span the full 2001-2024 range:

| Variable group | Available from |
|---|---|
| Air quality (NO2, O3, etc.) | 2001 |
| Traffic variables | 2015 |
| Meteorological variables | 2019 |

Analyses that involve meteorology or traffic are automatically restricted to the relevant time window. The forecasting model (Task 9) uses 2019-2024 because meteorological features are required as predictors.

### 5. Data quality issue categories

The pipeline checks for but does not automatically delete the following categories of problematic observations:

- Physically impossible values (e.g., `PRE = 0 hPa`, atmospheric pressure cannot be zero)
- Negative values for variables that cannot be negative (e.g., humidity, pollutant concentrations)
- Bounded-range violations (e.g., humidity outside 0-100%, wind direction outside 0-360 degrees)
- Extreme IQR outliers (k=3 per sensor). Note that many flagged values for NO and NOX are real pollution spikes, not sensor faults
- Duplicate (sensor, variable, timestamp) rows
- Inconsistent sensor coordinates across rows

These checks are used for diagnosis and interpretation in Task 2. They are flagged and reported, not silently removed.

### 6. Long sensor outages

Some sensors have multi-year consecutive gaps (e.g., Villaverde: approximately 3.5 years offline). Gaps of this length cannot be reliably handled by short-window temporal methods such as a 24-hour rolling mean. Spatial KNN imputation is more appropriate in these cases because it borrows from neighbouring stations at the same timestamp rather than from the sensor's own past.

### 7. Parallelisation notes

Task 8 runs independent per-year correlation jobs in parallel. The number of workers is controlled by `TASK8_MAX_WORKERS`. Set this to your machine's physical core count for best performance. Sequential and parallel outputs are verified using matrix hashes to confirm that parallelisation changes only runtime, not results.

---

## Limitations

- The pipeline is designed for offline batch processing. It is not suitable for real-time ingestion or streaming data.
- Imputation methods rely on spatial or temporal neighbours. Results may be less reliable for sensors with long simultaneous outages across the entire network.
- The forecasting model (Task 9) is limited to 2019-2024 due to meteorological feature availability. Results may not generalise to earlier periods.
- Traffic variables produced by RBF-Gaussian and RBF-Multiquadric interpolation show numerical instability and are excluded from predictive models.
- Correlation results for Task 6 and Task 8 reflect statistical association, not causal relationships.

---

## Citation

David Maria-Arribas et al. *METRAQ Air Quality dataset.* Hugging Face, 2024.
https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality
