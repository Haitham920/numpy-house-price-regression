# NumPy House Price Regression

An end-to-end house-price regressor built in *pure NumPy*, with no scikit-learn or pandas. It takes a raw numeric feature matrix that may contain missing values and outliers. It cleans the data, engineers a ratio feature and optional categorical encodings, splits the data reproducibly, standardizes it using training statistics only, fits an ordinary least squares (OLS) model, and reports MAE, RMSE, R² and residual statistics on held-out homes.

The project is built from 24 small, individually tested functions, each doing one step of a real ML workflow.

## Pipeline

raw X, y
  │
  ├─ 1. Clean       fill NaNs with column means → clip outliers to IQR bounds
  ├─ 2. Features    append a ratio feature → optionally append one-hot categories
  ├─ 3. Split       seeded shuffle → train / validation / test
  ├─ 4. Scale       mean/std from TRAIN only → applied to every split → bias column prepended
  ├─ 5. Fit         OLS on the training split (np.linalg.lstsq)
  └─ 6. Evaluate    MAE, RMSE, R², residual summary on validation and test

## Repository structure

| File | Contents |
|------|----------|
| model.py | The 24 pipeline functions, assembled from the step-by-step solutions |
| scaffold.py | Runnable demo of the full pipeline on synthetic housing data |
| docs/ | Supporting documentation |

## Getting started

Requirements: Python 3.9+ and NumPy.

git clone https://github.com/Haitham920/numpy-house-price-regression.git
cd numpy-house-price-regression
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install numpy
python scaffold.py

The demo generates 200 synthetic houses (rooms, households, house age, income, and a district label A/B/C), injects a few NaNs and outliers, runs the whole pipeline, and prints the test metrics alongside sample predictions.

## Usage

from model import house_price_pipeline

result = house_price_pipeline(
    X, y,
    ratio_num_idx=0,        # numerator column of the ratio feature
    ratio_den_idx=1,        # denominator column of the ratio feature
    cat_labels=districts,   # optional 1-D categorical array, or None
    train_ratio=0.7,
    val_ratio=0.15,         # the remainder becomes the test set
    seed=42,
    iqr_k=1.5,              # IQR multiplier for outlier clipping
)

result["theta"]          # fitted weights, shape (D,), bias first
result["y_test"]         # held-out targets
result["y_test_pred"]    # held-out predictions
result["test_metrics"]   # {'mae', 'rmse', 'r2', 'residual_summary'}
result["val_metrics"]    # same keys, computed on the validation split

## Function reference

### Data cleaning
| Function | Description |
|----------|-------------|
| impute_nan_with_mean(X) | Replaces NaNs with the column mean; all-NaN columns get 0 |
| compute_iqr_bounds(X, k) | Per-column bounds [Q1 − k·IQR, Q3 + k·IQR] |
| clip_columns(X, lower, upper) | Clips each value to its column's bounds |
| prepare_cleaned_features(X, iqr_k) | Impute, then clip |

### Feature engineering
| Function | Description |
|----------|-------------|
| make_ratio_feature(num, den, eps) | Safe ratio num / (den + eps) |
| append_column(X, col) | Appends one feature column |
| one_hot_encode(labels) | Dense binary matrix, one column per sorted category |
| assemble_feature_matrix(...) | Numeric features + ratio + optional one-hots |

### Splitting
| Function | Description |
|----------|-------------|
| make_shuffled_indices(n, seed) | Seeded permutation of row indices |
| partition_indices(idx, train_ratio, val_ratio) | Splits indices into train / val / test |
| subset_xy(X, y, idx) | Selects matching rows of X and y |
| make_train_val_test(X, y, ...) | Shuffles and returns the six split arrays in a dict |

### Scaling
| Function | Description |
|----------|-------------|
| fit_standardizer(X) | Per-column mean and std (zero std becomes 1) |
| apply_standardizer(X, mean, std) | (X − mean) / std |
| add_bias_column(X) | Prepends a column of ones |
| standardize_and_add_bias(splits) | Train-only standardization + bias for all splits |

### Model and evaluation
| Function | Description |
|----------|-------------|
| ols_fit(X, y) | Least-squares weights via np.linalg.lstsq |
| ols_predict(X, theta) | X @ theta |
| mean_absolute_error / root_mean_squared_error | MAE / RMSE |
| r_squared(y_true, y_pred) | R² (returns 0.0 when the target has no variance) |
| residual_summary(y_true, y_pred) | Mean, std, and median absolute residual |
| evaluate_predictions(y_true, y_pred) | All metrics in one dict |
| house_price_pipeline(...) | End-to-end entry point |

## Design notes

- *No leakage during scaling.* Standardization statistics come from the training split only and are reused for validation and test, as they would be for new houses at prediction time. (Imputation and IQR clipping run on the full matrix before splitting, as the exercise specifies.)
- **lstsq rather than solving the normal equation directly.** Solving XᵀX θ = Xᵀy breaks when features are collinear: a column that is a shifted copy of another, or a full one-hot block next to a bias column (the "dummy variable trap") make XᵀX singular. np.linalg.lstsq returns the minimum-norm least-squares solution and still recovers an exact fit when one exists.
- *Reproducible splits.* make_train_val_test uses NumPy's legacy seeding (np.random.seed + np.random.permutation) so that the permutation matches the reference results. The newer np.random.default_rng gives a different order for the same seed.
- *Tiny splits.* With only one validation sample, R² is undefined and is reported as 0.0.

## Possible improvements

- Vectorize the loop-based helpers (impute_nan_with_mean, clip_columns, one_hot_encode) with np.where, np.clip, and np.unique(..., return_inverse=True). This gives the same results much faster on large datasets.
- Make make_train_val_test and standardize_and_add_bias reuse the smaller helpers (make_shuffled_indices, partition_indices, fit_standardizer, ...) so that each piece of logic lives in one place. Use a local np.random.RandomState(seed) instead of reseeding the global generator.
- Use np.nanpercentile in compute_iqr_bounds so it stays correct if called on data that still contains NaNs.
- Fit the imputation and clipping statistics on the training split only.
- Add gradient descent and ridge regression as alternative solvers, and compare against scikit-learn's LinearRegression on the California Housing dataset.

## Author
**Haitham Maatar**
