#################################imposrting libraries
from scipy.spatial import ConvexHull
import time
import psutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
from numpy.linalg import inv
from pyDOE2 import lhs
import re
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import shutil
from scipy.spatial import ConvexHull
import sys
import matplotlib.lines as mlines
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.simplefilter('ignore')
#===================================datapoints transformation ====================================
def save_dataset(data, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    data.to_pickle(os.path.join(folder, filename))

# Transformations
def no_transform(points, *args):
    return points
    
def circle_transform(points, a, b):
    theta = 2 * np.pi * points[:, 0]
    r = np.abs(points[:, 1])  # taking absolute to ensure no negative values
    x_transformed = a * np.sqrt(r) * np.cos(theta)
    y_transformed = b * np.sqrt(r) * np.sin(theta)
    return np.column_stack((x_transformed, y_transformed))

def ellipse_transform(points, a, b):
    theta = 2 * np.pi * points[:, 0]
    r = np.abs(points[:, 1])  # taking absolute to ensure no negative values
    x_transformed = a * np.sqrt(r) * np.cos(theta)
    y_transformed = b * np.sqrt(r) * np.sin(theta)
    return np.column_stack((x_transformed, y_transformed))

def triangle_transform(points, x_min=-20, x_max=20, y_min=-20, y_max=20, std_scale=6.0):
    base = x_max - x_min
    height = y_max - y_min
    
    # Scale the points based on the desired standard deviation
    points_scaled = points * std_scale
    
    x_normalized = (points_scaled[:, 0] - points_scaled[:, 0].min()) / (points_scaled[:, 0].max() - points_scaled[:, 0].min())
    y_normalized = (points_scaled[:, 1] - points_scaled[:, 1].min()) / (points_scaled[:, 1].max() - points_scaled[:, 1].min())
    
    x_transformed = x_min + base * (1 - y_normalized) * (x_normalized - 0.5)
    y_transformed = y_min + height * y_normalized
    
    # Shifting the center of the triangle to (0, 0)
    x_center = (x_transformed.min() + x_transformed.max()) / 2
    y_center = (y_transformed.min() + y_transformed.max()) / 2
    x_transformed -= x_center
    y_transformed -= y_center
    
    return np.column_stack((x_transformed, y_transformed))


def extract_values_from_filename(filename):
    set_num = int(re.findall(r'df_(\d+)', filename)[0])
    std1 = float(re.findall(r'std1_([\d.]+)', filename)[0])
    dist1 = float(re.findall(r'dist1_([\d.]+)', filename)[0])
    ang1 = float(re.findall(r'ang1_([\d.]+)', filename)[0])
    std2 = float(re.findall(r'std2_([\d.]+)', filename)[0])
    dist2 = float(re.findall(r'dist2_([\d.]+)', filename)[0])
    ang2 = float(re.findall(r'ang2_([\d.]+)', filename)[0])
    return set_num, std1, dist1, ang1, std2, dist2, ang2
#===================================optional plotting of synthetic datasets generated for case2==========================
def plot_case2(df, folder, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(df[df['is_outlier'] == 0]['Feature 1'], df[df['is_outlier'] == 0]['Feature 2'], c='blue', label='Inliers', s=1)
    plt.scatter(df[(df['is_outlier'] == 1) & (df['outlier_set'] == 1)]['Feature 1'], df[(df['is_outlier'] == 1) & (df['outlier_set'] == 1)]['Feature 2'], c='red', label='Outliers Set 1', s=1, marker='^')
    plt.scatter(df[(df['is_outlier'] == 1) & (df['outlier_set'] == 2)]['Feature 1'], df[(df['is_outlier'] == 1) & (df['outlier_set'] == 2)]['Feature 2'], c='black', label='Outliers Set 2', s=1, marker='s')
    plt.legend()
    plt.title(filename)
    plt.xlabel('x')
    plt.ylabel('y')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

#===================================defining ground truth and overlap==========================
def gt_strict(X_datapoints, X_outliers1, X_outliers2):
    hull = ConvexHull(X_datapoints)
    is_outliers1_inside = False
    for outlier in X_outliers1:
        if np.all(hull.equations @ np.append(outlier, 1) <= 0):
            is_outliers1_inside = True
            break
    is_outliers2_inside = False        
    for outlier in X_outliers2:
        if np.all(hull.equations @ np.append(outlier, 1) <= 0):
            is_outliers2_inside = True
            break
    return not is_outliers1_inside, not is_outliers2_inside

def gt_relaxed(X_datapoints, outliers_centroid1, outliers_centroid2):
    hull = ConvexHull(X_datapoints)
    is_outliers1_outside = np.any(hull.equations @ np.append(outliers_centroid1, 1) > 0)
    is_outliers2_outside = np.any(hull.equations @ np.append(outliers_centroid2, 1) > 0)
    return is_outliers1_outside, is_outliers2_outside 

def overlap_between_outliers(X_outliers1, X_outliers2):
    hull1 = ConvexHull(X_outliers1)
    hull2 = ConvexHull(X_outliers2)
    overlap_1_in_2 = any(np.all(hull2.equations @ np.append(point, 1) <= 0) for point in X_outliers1)
    overlap_2_in_1 = any(np.all(hull1.equations @ np.append(point, 1) <= 0) for point in X_outliers2)

    if overlap_1_in_2 or overlap_2_in_1:
        return True 
    else:
        return False


#=================================== main analysis =================================== 

def cluster_mahala(XX, X, n_clusters, outlier_labels, compute_classification_metrics, random_seed=42, weight_thres=0.1, alpha_thres=0.05):
    # GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', init_params='kmeans', random_state=random_seed)
    gmm.fit(XX)
    cluster_labels = gmm.predict(XX)
    cluster_weights = gmm.weights_
    cluster_means = gmm.means_
    cluster_covariances = gmm.covariances_
    XX_values = XX.values  # Important to convert input DataFrame to numpy array to ensure further calculations run smoothly
    overall_mean = np.mean(XX_values, axis=0)
    overall_covariance = np.cov(XX_values, rowvar=False)

    # Calculating Mahalanobis distances and p-values
    mahalanobis_distances = [((cluster_mean - overall_mean).T @ inv(overall_covariance) @ (cluster_mean - overall_mean))
                            for cluster_mean in cluster_means]
    normalised_distances = (mahalanobis_distances - np.min(mahalanobis_distances)) / (np.max(mahalanobis_distances) - np.min(mahalanobis_distances))
    p_values = [chi2.sf(x**2, df=XX_values.shape[1]) for x in mahalanobis_distances]

    # Computing cluster characteristics and add all to DataFrame
    cluster_density = [np.sqrt(np.linalg.det(np.linalg.inv(cov))) / (2 * np.pi) for cov in cluster_covariances]
    cluster_std_dev = [np.sqrt(np.diag(cov)) for cov in cluster_covariances]
    cluster_stats_df = pd.DataFrame({
        'Cluster': range(n_clusters),
        'Mean1': cluster_means[:, 0],
        'Mean2': cluster_means[:, 1],
        'Variance1': [np.var(XX, axis=0)[0]] * n_clusters,
        'Variance2': [np.var(XX, axis=0)[1]] * n_clusters,
        'Weight': cluster_weights,
        'Density': cluster_density,
        'StdDev1': [std[0] for std in cluster_std_dev],
        'StdDev2': [std[1] for std in cluster_std_dev]
    })
    clusters_df = pd.DataFrame({'cluster_labels': cluster_labels, 'cluster_weight': cluster_weights[cluster_labels]})
    df_labeled = pd.concat([X, clusters_df], axis=1)
    X = df_labeled
    predicted_labels = np.isin(cluster_labels, np.where((cluster_weights <= weight_thres) & (np.array(p_values) <= alpha_thres)))
    classification_metrics = compute_classification_metrics(outlier_labels, predicted_labels)
    
    
    return X, XX_values,cluster_labels, cluster_weights, cluster_covariances, cluster_means, cluster_stats_df, clusters_df, classification_metrics,normalised_distances, p_values


#=================================== function to calculate purity metrics and plotting =================================== 
def compute_classification_metrics(ground_truth, predictions):
    # Initialize counters for TP, TN, FP, FN
    TP_p = TN_p = FP_p = FN_p = 0

    for true_label, predicted_label in zip(ground_truth, predictions):
        if true_label == 1:  # Outlier
            if predicted_label == 1:
                TP_p += 1  # True positive: correctly identified as outlier
            else:
                FN_p += 1  # False negative: should have been identified as outlier
        else:  # Not an outlier
            if predicted_label == 0:
                TN_p += 1  # True negative: correctly identified as not an outlier
            else:
                FP_p += 1  # False positive: incorrectly identified as outlier

    return {'TP_p': TP_p, 'TN_p': TN_p, 'FP_p': FP_p, 'FN_p': FN_p}



def compute_cluster_purity(n_clusters, outlier_labels, cluster_labels):
    """
    Computes purity for each cluster.

    Parameters:
    - n: Number of clusters
    - outlier_labels: List of labels indicating if a point is an outlier (1 for outlier, 0 otherwise)
    - cluster_labels: List of cluster labels for each point

    Returns:
    - cluster_purity: List of purity values for each cluster
    """  
    cluster_purity = []
    for i in range(n_clusters):
        cluster_points = outlier_labels[cluster_labels == i]
        purity = np.sum(cluster_points) / len(cluster_points)
        cluster_purity.append(purity)
        
    return cluster_purity

def compute_cluster_purity_both_out(n, outlier_labels, cluster_labels, outlier_set_labels):
    """
    Computes purity for each cluster for Outlier Set 1 and Outlier Set 2.

    Parameters:
    - n: Number of clusters.
    - outlier_labels: List of labels indicating if a point is an outlier (1 for outlier, 0 otherwise).
    - cluster_labels: List of cluster labels for each point.
    - outlier_set_labels: List indicating which set each outlier belongs to (1 for Outlier Set 1, 2 for Outlier Set 2).

    Returns:
    - cluster_purity_set1: List of purity values for each cluster for Outlier Set 1.
    - cluster_purity_set2: List of purity values for each cluster for Outlier Set 2.
    """
    cluster_purity_set1 = []
    cluster_purity_set2 = []

    for i in range(n):
        cluster_points = outlier_labels[cluster_labels == i] == 1 
        cluster_outlier_sets = outlier_set_labels[cluster_labels == i]

        # Calculate purity for Outlier Set 1
        purity_set1 = np.sum(cluster_points & (cluster_outlier_sets == 1)) / len(cluster_points)
        cluster_purity_set1.append(purity_set1)

        # Calculate purity for Outlier Set 2
        purity_set2 = np.sum(cluster_points & (cluster_outlier_sets == 2)) / len(cluster_points)
        cluster_purity_set2.append(purity_set2)

    return cluster_purity_set1, cluster_purity_set2


def get_purities_both(row):
    green_has_outlier1 = row['OS_1'] == 'yes'
    green_has_outlier2 = row['OS_2'] == 'yes'
    cluster_pvalues = row['p_values']
    outlier_labels = row['outlier_labels']
    cluster_labels = row['cluster_labels']

    green_clusters = [i for i, p in enumerate(cluster_pvalues) if p <= 0.05]

    outlier_cluster_purity = {'outlier_1': {}, 'outlier_2': {}}

    # Create dictionaries to store the count of outliers for each green cluster
    cluster_outlier_counts_1 = {cluster: 0 for cluster in green_clusters}
    cluster_outlier_counts_2 = {cluster: 0 for cluster in green_clusters}

    for i, label in enumerate(outlier_labels):
        cluster_label = cluster_labels[i]

        if cluster_label in green_clusters:
            if label == 1:  # Check if the data point belongs to outlier set 1
                cluster_outlier_counts_1[cluster_label] += 1
            elif label == 2:  # Check if the data point belongs to outlier set 2
                cluster_outlier_counts_2[cluster_label] += 1

    # Calculate the purity for each green cluster and outlier label
    for cluster_label in green_clusters:
        purity_1 = cluster_outlier_counts_1[cluster_label] / 75
        purity_2 = cluster_outlier_counts_2[cluster_label] / 75
        outlier_cluster_purity['outlier_1'][cluster_label] = purity_1
        outlier_cluster_purity['outlier_2'][cluster_label] = purity_2

    unique_cluster_purities_set1 = list(outlier_cluster_purity['outlier_1'].values())
    unique_cluster_purities_set2 = list(outlier_cluster_purity['outlier_2'].values())

    return unique_cluster_purities_set1, unique_cluster_purities_set2

def plot_purities_updated(dataframes_updated, output_folder_path):
    keys_order = ['Circle', 'Ellipse', 'Triangle', 'Irregular']
    subtitles = ['Case 2 (c)', 'Case 1(d)', 'Case 2(a)', 'Case 2(b)']
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs = axs.flatten()
    percentile_line_width = 5
    # Open a file to write the output
    with open(f'{output_folder_path}/purities_case2_results_all_required_subfolders.txt', 'w') as f:
        for idx, shape_name in enumerate(keys_order):
            ax = axs[idx]
            df = dataframes_updated[shape_name]
            all_data = []
            labels = []
            strict_TP_out1 = df[(df['classification_gts_outliers1'] == 'TP')]['outlier_1_purity_sum'].dropna().astype(float).tolist()
            strict_TP_out2 = df[(df['classification_gts_outliers2'] == 'TP')]['outlier_2_purity_sum'].dropna().astype(float).tolist()
            strict_TP = strict_TP_out1 + strict_TP_out2
            relaxed_TP_out1 = df[(df['classification_gtr_outliers1'] == 'TP')]['outlier_1_purity_sum'].dropna().astype(float).tolist()
            relaxed_TP_out2 = df[(df['classification_gtr_outliers2'] == 'TP')]['outlier_2_purity_sum'].dropna().astype(float).tolist()
            relaxed_TP = relaxed_TP_out1 + relaxed_TP_out2
            all_data.extend([strict_TP, relaxed_TP])
            labels.extend([f'Strict ({len(strict_TP)})', f'Relaxed ({len(relaxed_TP)})'])
            
            # Skip empty datasets
            if all(len(data) == 0 for data in all_data):
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set_title(f"{subtitles[idx]} {shape_name}", fontsize=16, weight='bold')
                continue
            
            bp = ax.boxplot(all_data, labels=labels, vert=True, patch_artist=True, notch=True,
                            boxprops=dict(facecolor="#ADD8E6"), medianprops=dict(color="black"),
                            whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                            flierprops=dict(markersize=4), whis=[10, 90])
            ax.set_ylabel('Purity', fontsize=16)
            ax.set_title(f"{subtitles[idx]} {shape_name}", fontsize=16, weight='bold')
            ax.set_xticklabels(labels, ha='center', fontsize=16)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            ax.set_ylim(0, 1)
            # Add 1st and 5th percentile lines
            for i, data in enumerate(all_data):
                if len(data) > 0:
                    fifth_percentile = np.percentile(data, 5)
                    ten_percentile = np.percentile(data, 10)
                    ax.plot([i + 1 - 0.2, i + 1 + 0.2], [fifth_percentile, fifth_percentile], color='red', linestyle='-', linewidth=8)
                    ax.plot([i + 1 - 0.2, i + 1 + 0.2], [ten_percentile, ten_percentile], color='blue', linestyle='-', linewidth=4)
            # Write sample sizes and average purities to the file
            f.write(f"Sample sizes and average purities for {shape_name}:\n")
            for label, data in zip(labels, all_data):
                if len(data) > 0:
                    f.write(f"{label}: Sample size = {len(data)}, Average purity = {np.mean(data):.2%}\n")
                else:
                    f.write(f"{label}: No data available\n")
        
        legend_lines = [
            plt.Line2D([0], [0], color='red', linestyle='-', linewidth=percentile_line_width, label="5th Percentile"),
            plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=4, label="10th Percentile")
        ]
        fig.legend(handles=legend_lines, loc="lower right", bbox_to_anchor=(1, -0.2), fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig(f'{output_folder_path}/purities_os2_updated.png', dpi=600, bbox_inches='tight')
        plt.close()

#=================================== gt classification calculations and plotting=================================== 


def classify_outliers(row):
    if row['OS_1'] == 'yes' and row['OS_2'] == 'yes':
        return 2
    elif row['OS_1'] == 'yes' or row['OS_2'] == 'yes':
        return 1
    else:
        return 0

def classify(row, ground_truth):
    if ground_truth in ['gts_outliers1', 'gtr_outliers1']:
        if row[ground_truth] == True and row['OS_1'] == 'yes':
            return 'TP'
        elif row[ground_truth] == False and row['OS_1'] == 'no':
            return 'TN'
        elif row[ground_truth] == False and row['OS_1'] == 'yes':
            return 'FP'
        elif row[ground_truth] == True and row['OS_1'] == 'no':
            return 'FN'
    
    if ground_truth in ['gts_outliers2', 'gtr_outliers2']:
        if row[ground_truth] == True and row['OS_2'] == 'yes':
            return 'TP'
        elif row[ground_truth] == False and row['OS_2'] == 'no':
            return 'TN'
        elif row[ground_truth] == False and row['OS_2'] == 'yes':
            return 'FP'
        elif row[ground_truth] == True and row['OS_2'] == 'no':
            return 'FN'






 

    


