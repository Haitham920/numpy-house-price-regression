"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X: np.ndarray) -> np.ndarray:
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """

   
    X_clean = X.copy()
    mean = np.nanmean(X_clean, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    for i in range(len(X_clean)):
        for j in range(len(X_clean[i])):
            if np.isnan(X_clean[i][j]):
                X_clean[i][j] = mean[j]
    return X_clean
    pass

# Step 2 - compute_iqr_bounds
import numpy as np

def compute_iqr_bounds(X, k=1.5):
    n, m = X.shape
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - (k * iqr)
    upper = q3 + (k * iqr)

    return (lower, upper)

# Step 3 - clip_columns
import numpy as np

def clip_columns(X, lower, upper):

    # Make a copy so the original X is not modified
    X_clean = X.copy()

    for i in range(len(X_clean)):
        for j in range(len(X_clean[i])):
            if X_clean[i][j] < lower[j]:
                X_clean[i][j] = lower[j]
            elif X_clean[i][j] > upper[j]:
                X_clean[i][j] = upper[j]

    return X_clean

# Step 4 - make_ratio_feature
import numpy as np
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return (numerator/(denominator+eps))
    pass

# Step 5 - append_column
import numpy as np
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X_copy=X.copy()
    NX=np.column_stack([X_copy,col])
    return(NX)
    pass

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    n=len(labels)
    c=len(np.unique(labels))
    binmat=np.zeros((n, c), dtype=float)
    for i in range (n):
        for j in range(c):
            if labels[i] == np.unique(labels)[j]:
                binmat[i][j] = 1.0
            else:
                binmat[i][j]=0.0
    return binmat
    pass

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features... 
    mean=np.mean(X, axis=0)
    std=np.std(X,axis=0)
    mask=np.where(std==0)[0]
    std[mask]=1.0
    return(mean,std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return ((X-mean)/std)

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    return(np.column_stack([np.ones((len(X)),),X]))
    pass

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng=np.random.default_rng(seed)
    idx=rng.permutation(n_samples)
    return(idx)
    pass

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

