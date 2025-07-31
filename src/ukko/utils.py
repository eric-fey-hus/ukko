# Define function to convert df into 3-D numpy array
import pandas as pd
import numpy as np

def convert_to_3d_df(df):
    """
    Convert a DataFrame with tuple column names ('feature', timepoint) to a 3D numpy array.

    Args:
        df (pd.DataFrame): Input DataFrame with columns as tuples ('feature', timepoint), 
        e.g. [('feature1', 0), ('feature1', 1), ..., ('featureN', T)]. 

    Returns:
        df_multiindex (pd.DataFrame): DataFrame with MultiIndex columns [Feature, Timepoint].
        data_3d (np.ndarray): Array of shape [samples, features, timepoints].
    """
    # Convert column names to tuples, assuming this "('feature', timepoint)"
    columns = [eval(col) for col in df.columns]
    df.columns = columns
    
    # Extract unique features and timepoints
    features = sorted(list(set([col[0] for col in columns])))
    timepoints = sorted(list(set([col[1] for col in columns])))
    
    # Initialize a 3D numpy array
    n_rows = df.shape[0]
    n_features = len(features)
    n_timepoints = len(timepoints)
    data_3d = np.empty((n_rows, n_features, n_timepoints))
    data_3d.fill(np.nan)
    
    # Map feature names and timepoints to indices
    feature_indices = {feature: i for i, feature in enumerate(features)}
    timepoint_indices = {timepoint: i for i, timepoint in enumerate(timepoints)}
    
    # Fill the 3D array with data from the DataFrame
    for col in columns:
        feature, timepoint = col
        feature_idx = feature_indices[feature]
        timepoint_idx = timepoint_indices[timepoint]
        data_3d[:, feature_idx, timepoint_idx] = df[col]

    # Create a MultiIndex for the columns of the 3D DataFrame
    columns = pd.MultiIndex.from_product([features, timepoints], names=["Feature", "Timepoint"])
    
    # Create the 3D DataFrame
    df_multiindex = pd.DataFrame(data_3d.reshape(n_rows, -1), columns=columns)
    
    return df_multiindex, data_3d


def downsample_timepoints(data_3f, rate=2):
    """
    Downsamples the timepoints of a 3D array (samples, features, timepoints) by the given rate.
    Each downsampled timepoint is:
      - the true non-nan value if only one exists,
      - the average of non-nan values if more than one,
      - nan if all are nan.

    Args:
        data_3f (np.ndarray): Array of shape (samples, features, timepoints)
        rate (int): Downsampling rate (e.g., 2 for half, 3 for third)

    Returns:
        np.ndarray: Downsampled array of shape (samples, features, timepoints // rate)
    """
    samples, features, timepoints = data_3f.shape
    new_timepoints = int(np.ceil(timepoints / rate))
    downsampled = np.full((samples, features, new_timepoints), np.nan, dtype=float)

    for t in range(new_timepoints):
        start = t * rate
        end = min((t + 1) * rate, timepoints)
        window = data_3f[:, :, start:end]  # shape: (samples, features, window_size)
        # Compute per sample-feature downsampled value
        for i in range(samples):
            for j in range(features):
                vals = window[i, j, :]
                not_nan = vals[~np.isnan(vals)]
                if len(not_nan) == 1:
                    downsampled[i, j, t] = not_nan[0]
                elif len(not_nan) > 1:
                    downsampled[i, j, t] = np.mean(not_nan)
                # else: remains nan
    return downsampled