#################################importing libraries
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
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.simplefilter('ignore')
###############################datapoints transformation ####################

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
#=========================================================Ground truth=========================================================================================

def gt_strict(X_datapoints, X_outliers):
    hull = ConvexHull(X_datapoints)
    for outlier in X_outliers:
        # Check if outlier overlaps with X_datapoints
        if any(np.all(outlier == datapoint) for datapoint in X_datapoints):
            continue  
        if all(hull.equations @ np.append(outlier, 1) <= 0):
            return False  # If any outlier is inside, return False

    return True  # If no outlier is inside, return True

def gt_relaxed(X_datapoints, outliers_centroid):
    hull = ConvexHull(X_datapoints)
    is_outside_hull = any(hull.equations @ np.append(outliers_centroid, 1) > 0)
    return is_outside_hull


#======================================================Optional plotting synthetic dataset generated===========================================================

def plot_case1(df, folder, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(df[df['is_outlier'] == False]['Feature 1'], df[df['is_outlier'] == False]['Feature 2'], c='blue', label='Inliers', s=1)
    plt.scatter(df[df['is_outlier'] == True]['Feature 1'], df[df['is_outlier'] == True]['Feature 2'], c='red', label='Outliers', s=1)
    plt.legend()
    plt.title(filename)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

#======================================================Main analysis========================================================================================

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
    p_values = [chi2.sf(x, df=XX_values.shape[1]) for x in mahalanobis_distances]

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

    return X, XX_values,cluster_labels, cluster_weights, cluster_covariances, cluster_means, cluster_stats_df, clusters_df, classification_metrics, normalised_distances, p_values

#=========================================================fOR PURITY calculations and plotting =============================================================

def compute_classification_metrics(ground_truth, predictions):
    TP_p = TN_p = FP_p = FN_p = 0
    for true_label, predicted_label in zip(ground_truth, predictions):
        if true_label == 1:  # Outlier
            if predicted_label == 1:
                TP_p += 1  # True positive
            else:
                FN_p += 1  # False negative
        else:  # Not an osti
            if predicted_label == 0:
                TN_p += 1  # True negative
            else:
                FP_p += 1  # False positive

    return {'TP_p': TP_p, 'TN_p': TN_p, 'FP_p': FP_p, 'FN_p': FN_p}

def compute_cluster_purity(outlier_labels, cluster_labels, n_clusters):
    cluster_purity = []
    for i in range(n_clusters):
        cluster_points = outlier_labels[cluster_labels == i]
        purity = np.sum(cluster_points) / len(cluster_points)
        cluster_purity.append(purity)
    return cluster_purity

def get_purities(row):
    cluster_purities = row['cluster_purity']
    p_values = row['p_values']

    outlier_cluster_indices = [i for i, p_value in enumerate(p_values) if p_value <= 0.05]
    outlier_cluster_purities = [cluster_purities[i] for i in outlier_cluster_indices]
    
    return outlier_cluster_purities


def process_dataframe_purity(df):
    df['Purity_TP'] = df['TP_p'] / 150
    return df

def plot_purities_updated(dataframes_updated, output_folder_path):
    keys_order = ['Circle', 'Ellipse', 'Triangle', 'Irregular']
    subtitles = ['Case 1 (a)', 'Case 1(b)', 'Case 1(a)', 'Case 1(b)']
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs = axs.flatten()
    percentile_line_width = 5
    
    # Open a file to write the output
    with open(f'{output_folder_path}/purities_case1_output.txt', 'w') as f:
        for idx, shape_name in enumerate(keys_order):
            ax = axs[idx]
            df = dataframes_updated[shape_name]
            all_data = []
            labels = []
            strict_TP = df[(df['Strict'] == 'TP')]['Purity_TP'].tolist()
            relaxed_TP = df[(df['Relaxed'] == 'TP')]['Purity_TP'].tolist()
            all_data.extend([strict_TP, relaxed_TP])
            labels.extend(['Strict', 'Relaxed'])
            bp = ax.boxplot(all_data, labels=labels, vert=True, patch_artist=True, notch=True,
                            boxprops=dict(facecolor="#ADD8E6"), medianprops=dict(color="black"),
                            whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                            flierprops=dict(markersize=4), whis=[10, 90])

            ax.set_ylabel('Purity', fontsize=16)
            ax.set_title(f"{subtitles[idx]} {shape_name}", fontsize=16, weight='bold')

            # Update x-tick labels to include counts
            xtick_labels = [f"{label} ({len(data)})" for label, data in zip(labels, all_data)]
            ax.set_xticklabels(xtick_labels, fontsize=16)

            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            ax.set_ylim(0, 1)
            # Add 1st and 5th percentile lines
            for i, data in enumerate(all_data):
                fifth_percentile = np.percentile(data, 5)
                first_percentile = np.percentile(data, 1)
                ten_percentile = np.percentile(data, 10)
                ax.plot([i + 1 - 0.2, i + 1 + 0.2], [fifth_percentile, fifth_percentile], color='red', linestyle='-', linewidth=8)
                ax.plot([i + 1 - 0.2, i + 1 + 0.2], [ten_percentile, ten_percentile], color='blue', linestyle='-', linewidth=4)
            # Print sample sizes and average purities
            f.write(f"Sample sizes and average purities for {shape_name}:\n")
            for label, data in zip(labels, all_data):
                f.write(f"{label}: Sample size = {len(data)}, Average purity = {np.mean(data):.2%}\n")
        
        legend_lines = [
            plt.Line2D([0], [0], color='red', linestyle='-', linewidth=percentile_line_width, label="5th Percentile"),
            plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=4, label="10th Percentile")
        ]
        fig.legend(handles=legend_lines, loc="lower right", bbox_to_anchor=(1, -0.2), fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig(f'{output_folder_path}/purities_os1_updated.png', dpi=600, bbox_inches='tight')
        plt.close()


#=========================================================EXTRACTING PARAMETERS=======================================================================

def extract_values_from_filename(filename):
    set_num = int(re.findall(r'df_(\d+)', filename)[0])
    clustd = float(re.findall(r'clustd_(\d+\.?\d*)', filename)[0])
    dist = float(re.findall(r'dist_(\d+\.?\d*)', filename)[0])
    angle = float(re.findall(r'angle_(\d+\.?\d*)', filename)[0])
    return set_num, clustd, dist, angle

#=============================================FOR CLASSSIFICATION AND F1 SCORE=============================================================

def classify(row, ground_truth):
    if row[ground_truth] == True and row['OSTI_identified'] == 'yes':
        return 'TP'
    elif row[ground_truth] == False and row['OSTI_identified'] == 'no':
        return 'TN'
    elif row[ground_truth] == False and row['OSTI_identified'] == 'yes':
        return 'FP'
    elif row[ground_truth] == True and row['OSTI_identified'] == 'no':
        return 'FN'

def plot_classifications(dataframes, classifications, color_mapping, output_folder_path):
    fig, axes = plt.subplots(2, len(dataframes) * len(classifications) // 2, figsize=(24, 12))
    subplot_titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for i, (label, df) in enumerate(dataframes.items()):
        for j, classification in enumerate(classifications):
            idx = i * len(classifications) + j
            row = idx // (len(dataframes) * len(classifications) // 2)
            col = idx % (len(dataframes) * len(classifications) // 2)
            ax = axes[row, col]
            sns.scatterplot(data=df, x='dist', y='angle', hue=classification, ax=ax, palette=color_mapping, s=60, edgecolor=(1, 1, 1, 0.5))
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 360])
            ax.set_yticks(range(0, 361, 60))
            ax.set_xlabel('Distance', fontdict={'fontsize': 18, 'family': 'Calibri'})
            if col == 0:
                ax.set_ylabel('Angle', fontdict={'fontsize': 18, 'family': 'Calibri'})
            else:
                ax.set_ylabel('')
            ax.set_title(f'{subplot_titles[idx]} {label}-{classification}', fontdict={'fontsize': 24, 'fontweight': 'bold', 'family': 'Calibri'})
            ax.get_legend().remove()

    legend_handles = [mpatches.Patch(color=color, label=class_label) for class_label, color in color_mapping.items()]
    legend = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=len(color_mapping), fontsize=16, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{output_folder_path}/scatterplots_case1.png', dpi=600)
    plt.close()

    # Calculate and save classification metrics to a file
    with open(f'{output_folder_path}/classification_metrics_case1.txt', 'w') as f:
        for label, df in dataframes.items():
            f.write(f"Metrics for {label}:\n")
            for classification in classifications:
                TP = sum(df[classification] == 'TP')
                FP = sum(df[classification] == 'FP')
                FN = sum(df[classification] == 'FN')
                TN = sum(df[classification] == 'TN')
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f.write(f"{classification} - TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score {f1_score:.2f}\n")

#=========================================================================================================================================================