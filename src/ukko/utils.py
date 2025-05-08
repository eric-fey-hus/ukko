# Define function to convert df into 3-D numpy array
import pandas as pd
import numpy as np

def convert_to_3d_df(df):

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
