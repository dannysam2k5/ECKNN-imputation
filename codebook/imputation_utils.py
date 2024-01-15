# imputation_utils.py

import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error

def calculate_avg_similarity_matrix(scaled_features_df, num_iterations):
    """
    Calculate the average similarity matrix using KMeans clustering.

    Parameters:
    scaled_features_df (pandas.DataFrame): DataFrame containing scaled features.
    num_iterations (int, optional): Number of iterations for the clustering process. Default is 10.

    Returns:
    avg_similarity_matrix (numpy.ndarray): Average similarity matrix.
    
    Usage Example:
    avg_similarity_matrix = calculate_avg_similarity_matrix(scaled_features_df, num_iterations)
    """

    # Extract scaled features
    scaled_features_np = scaled_features_df.to_numpy()

    # Get Number of Samples
    num_samples = scaled_features_np.shape[0]

    # Initialize average similarity matrix
    avg_similarity_matrix = np.zeros((num_samples, num_samples))

    # Iterate multiple times
    for i in range(num_iterations):
        # select random k value
        k = random.randint(2, 20)
#         print(f"Iteration {i+1}, k = {k}")

        # Create K Means model
        kmeans_model = KMeans(n_clusters=k)
        cluster_labels = kmeans_model.fit_predict(scaled_features_np)

        # Calculate pairwise similarity matrix
        similarity_matrix = np.equal.outer(cluster_labels, cluster_labels).astype(int)

        # Update average similarity matrix
        avg_similarity_matrix += similarity_matrix

    # Take average of similarity matrices
    avg_similarity_matrix /= num_iterations

    return avg_similarity_matrix


def artificial_missingness(dataset, missing_rate):
    """
    Introduce missing values artificially into a dataset following a MCAR (Missing Completely At Random) pattern.

    Parameters:
    dataset (pandas.DataFrame): Input dataset.
    missing_rate (float, optional): Proportion of missing values to introduce.

    Returns:
    missing_dataset (pandas.DataFrame): Dataset with introduced missing values.
    
    Usage Example:
    missing_dataset = artificial_missingness(processed_dataset, missing_rate=0.10)
    """
    
    # Make a copy of the input dataset
    missing_dataset = dataset.copy()

    # Extract the number of records and features
    num_rows, num_cols = missing_dataset.shape

    num_missing_values = int(num_rows * num_cols * missing_rate)

    # select positions randomly and Perform MCAR missingness
    positions = np.random.choice(num_rows * num_cols, num_missing_values, replace=False)

    for pos in positions:
        row_idx = pos // num_cols
        col_idx = pos % num_cols
        missing_dataset.iat[row_idx, col_idx] = np.nan

    return missing_dataset
    
def feature_average_imputation(dataset):
    """
    Impute missing values in a dataset using the average value of each feature.

    Parameters:
    dataset (pandas.DataFrame): Input dataset with missing values.

    Returns:
    avg_imputed_dataset (pandas.DataFrame): Dataset with missing values imputed using feature averages.
    
    Usage Example:
    avg_imputed_dataset = feature_average_imputation(missing_dataset)
    print("\nDataFrame after feature Imputation:")
    print(avg_imputed_dataset)
    """
        
    # Make a copy of the input dataset
    avg_imputed_dataset = dataset.copy()

    # Impute missing values using feature average
    for feat_col in avg_imputed_dataset.columns:
        feat_col_mean = avg_imputed_dataset[feat_col].mean()
        avg_imputed_dataset[feat_col].fillna(feat_col_mean, inplace=True)

    return avg_imputed_dataset
    
def ecknn_imputation(missing_dataset, avg_imputed_dataset, avg_similarity_matrix, k_neighbors):
    """
    Impute missing values using Enhanced k-Nearest Neighbors (EckNN) imputation method.

    Parameters:
    missing_dataset (pandas.DataFrame): Input dataset with missing values.
    avg_imputed_dataset (pandas.DataFrame): Dataset with missing values imputed using feature averages.
    avg_similarity_matrix (numpy.ndarray): Average similarity matrix.
    k_neighbors (int, optional): Number of nearest neighbors to consider. Default is 3.

    Returns:
    ecknn_imputed_dataset (numpy.ndarray): Dataset with missing values imputed using EckNN method.
    
    Usage Example:
    ecknn_imputed_dataset = ecknn_imputation(missing_dataset, avg_imputed_dataset, avg_similarity_matrix, k_neighbors)
    """
    
    # Make a copy of the input datasets
    ecknn_imputed_dataset = missing_dataset.copy()
    ecknn_avg_imputed_dataset = avg_imputed_dataset.copy()

    # Convert the dataframes to numpy arrays
    ecknn_imputed_dataset = ecknn_imputed_dataset.to_numpy()
    ecknn_avg_imputed_dataset = ecknn_avg_imputed_dataset.to_numpy()

    # Get Number of Samples and features
    num_samples, num_features = ecknn_imputed_dataset.shape

    # Exclude self similarity by setting the diagonal to 0
    np.fill_diagonal(avg_similarity_matrix, 0)

    for i in range(num_samples):
        missing_indices = np.where(np.isnan(ecknn_imputed_dataset[i]))[0]

        if missing_indices.size > 0:
            for col_idx in missing_indices:
                # Find the k-nearest neighbors based on the similarity matrix
                similarity_row = avg_similarity_matrix[i]
                # Sort in descending order
                sorted_indices = np.argsort(similarity_row)[::-1]
                                               
                # Take top k max value similarity indices
                k_max_indices = sorted_indices[:k_neighbors]            

                if k_neighbors > 1:
                    # Compute Euclidean distance and select best k_neighbors-value
                    distances = euclidean_distances([ecknn_avg_imputed_dataset[i]],
                                                    ecknn_avg_imputed_dataset[sorted_indices[:k_neighbors]])

                    # Take indices with minimum distance
                    nn_idx = sorted_indices[np.argsort(distances)][0:k_neighbors]

                    # use nn_idx (minimum distance) for imputation
                    ecknn_imputed_dataset[i, col_idx] = np.nanmean(ecknn_avg_imputed_dataset[nn_idx, col_idx])
                else:
                    # use k_max_indices for imputation
                    # Set k_neighbors = 1
                    ecknn_imputed_dataset[i, col_idx] = ecknn_avg_imputed_dataset[k_max_indices, col_idx]

    return ecknn_imputed_dataset
    
def calculate_nrmse(processed_dataset, missing_dataset, ecknn_imputed_dataset):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between imputed and true values.

    Parameters:
    processed_dataset (pandas.DataFrame): Complete dataset without missing values.
    missing_dataset (pandas.DataFrame): Input dataset with missing values.
    ecknn_imputed_dataset (numpy.ndarray): Dataset with missing values imputed using EckNN method.

    Returns:
    nrmse (float): Normalized Root Mean Squared Error.
    
    Usage Example:
    nrmse = calculate_nrmse(processed_dataset, missing_dataset, ecknn_imputed_dataset)
    print("NRMSE:", nrmse)
    """
    
    # Find indices of missing values
    missing_indices = np.isnan(missing_dataset.to_numpy())

    # Calculate NRMSE between the complete data and imputed data for missing values
    true_values = processed_dataset.to_numpy()[missing_indices]
    imputed_values = ecknn_imputed_dataset[missing_indices]

    nrmse = np.sqrt(mean_squared_error(true_values, imputed_values)) / (true_values.max() - true_values.min())

    return nrmse