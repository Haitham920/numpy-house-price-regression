"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np
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

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n=indices.shape[0]
    n_train= int(n * train_ratio)
    n_val=int( n * val_ratio)
    train_idx, val_idx, test_idx=indices[:n_train] , indices[n_train:n_train+n_val] , indices[n_train+n_val:]
    return train_idx,val_idx,test_idx
    pass

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return(X[indices],y[indices])

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A=X.T @ X
    b=X.T @ y
    return(np.linalg.solve(A,b))
    pass

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    y_pred= X @ theta
    return y_pred
    pass

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return((1/len(y_true)*np.sum(np.abs(y_true-y_pred))))
    pass

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return (np.sqrt((1/len(y_true))*np.sum((y_true-y_pred)**2)))
    pass

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    sst=np.sum((y_true-np.mean(y_true, axis=0))**2)
    if sst==0:
        return(0.0)
    else:    
        return(1-(np.divide((np.sum((y_true-y_pred)**2)), (np.sum((y_true-np.mean(y_true, axis=0))**2)))))

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    errors=y_true-y_pred
    return {'mean':np.mean(errors),'std':np.std(errors),'median_abs':np.median(np.abs(errors))}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    imp=impute_nan_with_mean(X)
    iqr=compute_iqr_bounds(imp, iqr_k)
    b=clip_columns(imp, iqr[0], iqr[1])
    return b

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # 1. Extract 1-D feature columns using 2D slicing [:, col_idx]
    num = X_num[:, ratio_num_idx]
    den = X_num[:, ratio_den_idx]
    
    # 2. Compute ratio and append as a new column
    mrf = make_ratio_feature(num, den, eps=1e-8)
    a = append_column(X_num, mrf)
    
    # 3. Horizontally stack one-hot encoded matrix if categorical labels are provided
    if cat_labels is not None:
        b = one_hot_encode(cat_labels)
        a = np.hstack([a, b])  # Or np.concatenate([a, b], axis=1)
        
    return a

# Step 21 - make_train_val_test
import numpy as np

def make_train_val_test(X, y, train_ratio, val_ratio, seed=None):
    n_samples = len(X)
    
    # Legacy NumPy seeding to match expected permutation sequence
    if seed is not None:
        np.random.seed(seed)
        
    shuffled_indices = np.random.permutation(n_samples)
    
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    return {
        'X_train': X_shuffled[:train_end],
        'y_train': y_shuffled[:train_end],
        'X_val': X_shuffled[train_end:val_end],
        'y_val': y_shuffled[train_end:val_end],
        'X_test': X_shuffled[val_end:],
        'y_test': y_shuffled[val_end:]
    }

# Step 22 - standardize_and_add_bias
import numpy as np

def standardize_and_add_bias(splits, eps=1e-8):
    X_train = splits['X_train']
    
    # 1. Compute mean and standard deviation from X_train
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # 2. Handle constant features (std == 0) by replacing 0.0 with 1.0
    std_adjusted = std.copy()
    std_adjusted[std_adjusted < eps] = 1.0
    
    # Copy dictionary structure
    std_splits = splits.copy()
    
    # 3. Transform features and prepend bias column
    for key in ['X_train', 'X_val', 'X_test']:
        if key in splits:
            X = splits[key]
            
            # Standardize using adjusted std
            X_std = (X - mean) / std_adjusted
            
            # Prepend bias column of ones
            bias_col = np.ones((X_std.shape[0], 1), dtype=X_std.dtype)
            std_splits[key] = np.hstack([bias_col, X_std])
            
    # Return std with zero-std values replaced by 1.0 to match expected output
    return std_splits, mean, std_adjusted

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

