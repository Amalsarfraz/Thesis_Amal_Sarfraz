#!/usr/bin/env python
# coding: utf-8

# ## This routine is for generating synthetic datasets for case 1 and analsyisng them. All the figures added in paper titled "A Robust Two-step Method for Clustering and Dealing with Outlier Sets in Big Data". For questions contact asarfraz1@sheffield.ac.uk 

# ## Importing libararies


import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
import time
import psutil
import pandas as pd
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
from numpy.linalg import inv
from pyDOE2 import lhs
import re
import matplotlib.patches as mpatches
import shutil
import sys
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.simplefilter('ignore')


# In[5]:


from case1 import save_dataset,no_transform,circle_transform,ellipse_transform,triangle_transform,plot_case1,gt_strict,gt_relaxed,cluster_mahala,compute_classification_metrics,compute_cluster_purity,extract_values_from_filename,classify,get_purities,process_dataframe_purity,plot_purities_updated,plot_classifications


# In[6]:


sns.set_context('talk')
sns.set(rc={
    "axes.labelsize": 18,     
    "axes.titlesize": 18,        
    "xtick.labelsize": 14,       
    "ytick.labelsize": 14,      
    "legend.fontsize": 24,       
    "axes.facecolor": "white",   
    "figure.facecolor": "white"  
})
sns.set_style("white")


# ## Synthetic Dataset Generation for Case 1 (One Outlier Set)

# In[7]:


##############################defining parameters ##########################################
# Generate and save datasets
transform_funcs = {
    "circle": (circle_transform, (5, 5)),
    "ellipse": (ellipse_transform, (5, 10)),
    "triangle": (triangle_transform, (-20, 20, -20, 20,6)),
    "no_transform": (no_transform, ())
}

n_datasets = 4000 # Total datasets
n_per_shape = 1000  # Datasets per shape
n_datapoints = 1500
n_outliers = 150
random_state=42

cluster_std_range = [1, 10]
desired_distance_range = [0, 100]
desired_angle_range = [0, 360]

samples = lhs(3, samples=n_datasets)
cluster_std_samples = cluster_std_range[0] + samples[:, 0] * (cluster_std_range[1] - cluster_std_range[0])
desired_distance_samples = desired_distance_range[0] + samples[:, 1] * (desired_distance_range[1] - desired_distance_range[0])
desired_angle_samples = desired_angle_range[0] + samples[:, 2] * (desired_angle_range[1] - desired_angle_range[0])

shape_transforms = ['circle', 'ellipse', 'triangle', 'no_transform']


# In[8]:


for i, (cluster_std, desired_distance, desired_angle) in enumerate(zip(cluster_std_samples, desired_distance_samples, desired_angle_samples)):
    X_datapoints, _ = make_blobs(n_samples=n_datapoints, centers=15, cluster_std=6, center_box=(-10, 10))
    inliers = X_datapoints.copy()
    is_outlier = np.full((X_datapoints.shape[0], 1), False)
    labels = np.full((X_datapoints.shape[0], 1), 0)

    datapoints_centroid = np.mean(X_datapoints, axis=0)
    angle_rad = np.deg2rad(desired_angle)
    outliers_centroid = np.array([datapoints_centroid[0] + desired_distance * np.cos(angle_rad), datapoints_centroid[1] + desired_distance * np.sin(angle_rad)])

    X_outliers, _ = make_blobs(n_samples=n_outliers, centers=1, cluster_std=cluster_std, random_state=random_state)
    X_outliers += outliers_centroid - np.mean(X_outliers, axis=0)
    outliers = X_outliers.copy()
    is_outlier_outliers = np.full((n_outliers, 1), True)
    labels_outliers = np.full((n_outliers, 1), 1)

    dataset = np.vstack([inliers, outliers])
    is_outlier = np.vstack([is_outlier, is_outlier_outliers])
    labels = np.vstack([labels, labels_outliers])

    shape_index = i // n_per_shape
    transform_name = shape_transforms[shape_index]
    transform_func, transform_params = transform_funcs[transform_name]
    transformed_inliers = transform_func(inliers, *transform_params)

    # Creating final dataset
    transformed_data = np.vstack([transformed_inliers, outliers])
    is_outlier = np.vstack([np.zeros((n_datapoints, 1), dtype=bool), np.ones((n_outliers, 1), dtype=bool)])
    labels = np.vstack([np.zeros((n_datapoints, 1), dtype=int), np.ones((n_outliers, 1), dtype=int)])
    final_dataset = np.hstack([transformed_data, is_outlier, labels])

    df = pd.DataFrame(final_dataset, columns=["Feature 1", "Feature 2", "is_outlier", "label"])
    filename = f"df_{i}_clustd_{round(cluster_std, 2)}_dist_{round(desired_distance, 2)}_angle_{round(desired_angle, 2)}.pkl"
    save_dataset(df, f"../results/Case1_datasets/{transform_name}", filename)

    # Save outlier centroid to a .txt file
    centroid_filename = f"df_{i}_clustd_{round(cluster_std, 2)}_dist_{round(desired_distance, 2)}_angle_{round(desired_angle, 2)}_centroid.txt"
    centroid_filepath = os.path.join(f"../results/Case1_datasets/{transform_name}", centroid_filename)
    np.savetxt(centroid_filepath, outliers_centroid, delimiter=',')
    
#print("All Case 1 datasets were generated successfully.")


# *Plotting of the datasets is optional and would required 35-45 minutes*

# In[9]:


# #=================Optional plotting for synthetic datasets generated
# for transform_name in ["circle", "ellipse", "triangle", "no_transform"]:
#     transform_folder = os.path.join('Case1_datasets', transform_name)
#     for filename in os.listdir(transform_folder):
#         if filename.endswith(".pkl"):
#             df = pd.read_pickle(os.path.join(transform_folder, filename))
#             plot_filename = filename.replace('.pkl', '.png')
#             plot_folder = os.path.join(transform_folder, 'plots')
#             plot_case1(df, plot_folder, plot_filename)

# print("All Case 1 datasets and plotted successfully.")


# ## Running OSTI for Case 1

# *The analysis below takes 40 minutes on average to run for each requried_subfolders, based on your hardware specififcations please run for each folder iteratievly or on all folders automatically*

# In[10]:


base_folder = '../results/Case1_datasets'


# In[11]:


#for running one folder at a time uncomment the below lines (40 minutes on average for each folder)
# # Ensure to have all the shapes results run before heading to the analysis bit 
#required_subfolders = {'circle'}
#required_subfolders = {'ellipse'}
#required_subfolders = {'triangle'}
#required_subfolders = {'no_transform'}


# In[12]:


#for running all folders (will take approx 2-3 hours)
required_subfolders = {'circle','ellipse','triangle', 'no_transform'}


# In[13]:


for shape in required_subfolders:
    shape_path = os.path.join(base_folder, shape)
    if not os.path.exists(shape_path):
        print(f"Shape subfolder not found: {shape_path}")
        continue

    pkl_files = glob.glob(os.path.join(shape_path, '*.pkl'))
    OSTI_results = []
    gt1_results = []
    gt2_results = []

    start_wall_time = time.time()
    start_cpu_time = time.process_time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    
    for pkl_file in pkl_files:
        data = pd.read_pickle(pkl_file)
        filename = os.path.basename(pkl_file).replace('.pkl', '')
        transform_name = shape
        results_subfolder = os.path.join(shape_path, f'results_{shape}')
        os.makedirs(results_subfolder, exist_ok=True)

        centroid_filename = f"{filename}_centroid.txt"
        centroid_filepath = os.path.join(shape_path,centroid_filename)
        if os.path.exists(centroid_filepath):
            with open(centroid_filepath, 'r') as f:
                centroid_data = f.read().strip()
                centroid_values = re.findall(r'[-+]?\d+\.?\d*[eE]?[-+]?\d*', centroid_data)
                expected_outliers_centroid = np.array([float(val) for val in centroid_values])
        else:
            print(f"Centroid file not found: {centroid_filepath}. Skipping...")
            expected_outliers_centroid = None

        XX = data
        X = data[['Feature 1', 'Feature 2']]
        outlier_labels = data['is_outlier']
        n_clusters = 8
############################################################################
        start_wall_time1 = time.time()
        start_cpu_time1 = time.process_time()
        process1 = psutil.Process(os.getpid())
        start_memory1 = process.memory_info().rss


        (X, XX_values, cluster_labels, cluster_weights, cluster_covariances, cluster_means,
         cluster_stats_df, clusters_df, classification_metrics, normalised_distances,
         p_values) = cluster_mahala(X, X, n_clusters, outlier_labels,
                                    compute_classification_metrics, random_seed=42,
                                    weight_thres=0.1, alpha_thres=0.05)

        end_wall_time1 = time.time()
        end_cpu_time1 = time.process_time()
        end_memory1 = process.memory_info().rss
        wall_time1 = end_wall_time1 - start_wall_time1
        cpu_time1 = end_cpu_time1 - start_cpu_time1
        memory_used1 = end_memory1 - start_memory1


        with open(os.path.join(results_subfolder, f'{filename}_{transform_name}_results.txt'), 'w') as f:
            f.write(f'The script took {wall_time1} seconds (wall clock time) to run.\n')
            f.write(f'The script took {cpu_time1} seconds of CPU time to run.\n')
            f.write(f'The script used {memory_used1} bytes of memory.\n')
#################################################################################
        cluster_purity = compute_cluster_purity(outlier_labels, cluster_labels, n_clusters)

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        outliers = XX[XX['is_outlier'] == 1]
        non_outliers = XX[XX['is_outlier'] == 0]
        axes[0].scatter(non_outliers['Feature 1'], non_outliers['Feature 2'], s=60, alpha=0.7, label='datapoints', marker='o')
        axes[0].scatter(outliers['Feature 1'], outliers['Feature 2'], s=60, alpha=0.5, label='Outliers', marker='^', color='red')
        axes[0].set_title(f'Clustering Result', weight='bold')
        axes[0].set_xlabel('Feature 1', weight='bold')
        axes[0].set_ylabel('Feature 2', weight='bold')
        axes[0].legend(bbox_to_anchor=(1.2, 1), title="", loc='upper right')

        outlier_points = XX_values[outlier_labels == 1]
        axes[1].scatter(outlier_points[:, 0], outlier_points[:, 1], s=50, alpha=0.4, label='Outliers', marker='^')
        palette = sns.color_palette('husl', n_clusters)
        for i in range(n_clusters):
            cluster_points = XX_values[cluster_labels == i]
            color = palette[i]
            axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=color, alpha=0.7, label=f'Cluster {i} Weight={cluster_weights[i]:.2f}')
            axes[1].text(cluster_means[i, 0], cluster_means[i, 1], str(i), color=color, ha='center', va='center', fontsize=12, weight='bold', bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.5'))
            if cluster_weights[i] <= 0.1:
                for point in cluster_points:
                    axes[1].plot(point[0], point[1], marker='+', markersize=5, color="black")
        axes[1].set_title('Candidate OSTI', weight='bold')
        axes[1].set_xlabel('Feature 1', weight='bold')
        axes[1].set_ylabel('Feature 2', weight='bold')
        axes[1].legend(bbox_to_anchor=(1.2, 1.2), title="Clusters and their weights", loc='upper right')

        highlight_green = [True if w <= 0.1 and p <= 0.05 else False for w, p in zip(cluster_weights, p_values)]
        colors = ['green' if h_green else 'blue' for h_green in highlight_green]
        sizes = [400 if h_green else 250 for h_green in highlight_green]
        axes[2].scatter(normalised_distances, p_values, c=colors, s=sizes)
        padding = 0.05
        legend_elements_purity = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}, Purity: {cluster_purity[i]:.2f}', markerfacecolor='black', markersize=10) for i in range(n_clusters)]
        leg2 = axes[2].legend(handles=legend_elements_purity, loc='upper right', title="Cluster purities", bbox_to_anchor=(1.2, 1.2))
        axes[2].add_artist(leg2)
        axes[2].set_xlabel('Normalised Mahalanobis Distance', weight='bold')
        axes[2].set_ylabel('P-value', weight='bold')
        plt.yticks(np.arange(0, np.ceil(max(p_values)) + 0.1, step=0.1))
        plt.ylim(0 - padding, np.ceil(max(p_values)) + padding)
        axes[2].set_title('OSTI', weight='bold')

        for i, (normalised_distance, p_value) in enumerate(zip(normalised_distances, p_values)):
            axes[2].text(normalised_distance, p_value, f'{i}', fontsize=12, ha='center', va='center', color='white')
        axes[2].grid(True)
        fig.tight_layout()

        fig.savefig(os.path.join(results_subfolder, f"{filename}.png"), dpi=60, bbox_inches='tight')
        plt.close()

        has_green_cluster = any(color == 'green' for color in colors)
        cluster_info = clusters_df.to_dict('records')

        # Compute ground truth classification
        set_num, clustd, dist, angle = extract_values_from_filename(filename)

        datapoints_centroid = np.mean(non_outliers[['Feature 1', 'Feature 2']].values, axis=0)
        angle_rad = np.deg2rad(angle)
        if expected_outliers_centroid is not None:
            X_outliers = outliers[['Feature 1', 'Feature 2']].values
            #X_outliers += expected_outliers_centroid - np.mean(X_outliers, axis=0)
            X_non_outliers = non_outliers[['Feature 1', 'Feature 2']].values

            gt1 = gt_strict(X_non_outliers, X_outliers)
            gt2 = gt_relaxed(X_non_outliers, expected_outliers_centroid)

            gt1_results.append(gt1)
            gt2_results.append(gt2)
        else:
            gt1_results.append(None)
            gt2_results.append(None)

        OSTI_results.append({
            'filename': filename,
            'OSTI_identified': 'yes' if has_green_cluster else 'no',
            'set_num': set_num,
            'clustd': clustd,
            'dist': dist,
            'angle': angle,
            'cluster_purity': cluster_purity,
            'p_values': p_values,
            'outlier_labels': outlier_labels.tolist(),
            'cluster_info': cluster_info,
            **classification_metrics
        })

    OSTI_analysis = pd.DataFrame(OSTI_results)
    OSTI_analysis['gt1_strict'] = gt1_results
    OSTI_analysis['gt2_relaxed'] = gt2_results

    OSTI_analysis.to_pickle(os.path.join(results_subfolder, f'{transform_name}_OS1.pkl'))

    # End measurement
    end_wall_time = time.time()
    end_cpu_time = time.process_time()
    end_memory = process.memory_info().rss
    wall_time = end_wall_time - start_wall_time
    cpu_time = end_cpu_time - start_cpu_time
    memory_used = end_memory - start_memory

    # Save timing and memory usage
    with open(os.path.join(results_subfolder, f'{transform_name}_results.txt'), 'w') as f:
        f.write(f'The script took {wall_time} seconds (wall clock time) to run.\n')
        f.write(f'The script took {cpu_time} seconds of CPU time to run.\n')
        f.write(f'The script used {memory_used} bytes of memory.\n')

#print("All analysis on Case 1 datasets was conducted successfully.")


# ## Analysing results

# In[14]:


import os
import shutil
import glob

# Base directory containing the subfolders
#======================================
base_folder = '../results/Case1_datasets'
required_subfolders = {'circle', 'ellipse', 'triangle', 'no_transform'}  # Ensure all the results for each shape are present
new_analysis_folder = os.path.join(base_folder, 'analysis')
os.makedirs(new_analysis_folder, exist_ok=True)

# Iterate over the required subfolders and move .pkl files to the analysis folder
for shape in required_subfolders:
    shape_path = os.path.join(base_folder, shape)
    results_subfolder = os.path.join(shape_path, f'results_{shape}')

    if os.path.exists(results_subfolder):
        pkl_files = glob.glob(os.path.join(results_subfolder, '*.pkl'))
        
        for pkl_file in pkl_files:
            # Define the new path for the .pkl file in the analysis folder
            new_file_path = os.path.join(new_analysis_folder, os.path.basename(pkl_file))
            
            # Copy the file to the new directory
            shutil.copy(pkl_file, new_file_path)
    else:
        pass  # Results subfolder does not exist, skip to the next shape


# In[15]:


#==============================updated based on your path=========
folder_path = new_analysis_folder
#=================================================================

output_folder_path = '../results/Case1_datasets/Case1_FINAL_result_pickles'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

pickle_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

# Process each pickle file
for file in pickle_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_pickle(file_path)
    #applying classifications
    ground_truths = ['gt1_strict', 'gt2_relaxed']
    for ground_truth in ground_truths:
        classification_column = f'classification_{ground_truth}'
        df[classification_column] = df.apply(lambda row: classify(row, ground_truth), axis=1)

    df['outlier_cluster_purities'] = df.apply(get_purities, axis=1)
    
    base_filename = os.path.splitext(file)[0]  
    modified_filename = f"{base_filename}_results.pkl" 
    output_file_path = os.path.join(output_folder_path, modified_filename)
    df.to_pickle(output_file_path)

#print("Processing complete.")


# In[16]:


circle_filename = 'circle_OS1_results.pkl'
ellipse_filename = 'ellipse_OS1_results.pkl'
triangle_filename = 'triangle_OS1_results.pkl'
no_transform_filename = 'no_transform_OS1_results.pkl'

df_c = pd.read_pickle(os.path.join(output_folder_path, circle_filename))
df_e = pd.read_pickle(os.path.join(output_folder_path, ellipse_filename))
df_t = pd.read_pickle(os.path.join(output_folder_path, triangle_filename))
df_i = pd.read_pickle(os.path.join(output_folder_path, no_transform_filename))


# In[17]:


for df in [df_e, df_c,df_t,df_i]:
    df.rename(columns={
        'classification_gt1_strict': 'Strict',
        'classification_gt2_relaxed': 'Relaxed'
    }, inplace=True)


# In[18]:


dataframes_updated = {
   'Circle': df_c,
    'Ellipse': df_e,
    'Triangle': df_t,
    'Irregular': df_i,
}


# ## Classification scatter plots

# In[19]:


classifications = ['Strict', 'Relaxed']
color_mapping = {'TP': 'green', 'TN': '#56B4E9', 'FP': 'black', 'FN': '#F0E442'}

plot_classifications(dataframes_updated, classifications, color_mapping,output_folder_path)


# ## Purities

# In[20]:


for shape, df in dataframes_updated.items():
    dataframes_updated[shape] = process_dataframe_purity(df)


# ## Purities plotting code below

# In[21]:


plot_purities_updated(dataframes_updated,output_folder_path)


# ## Heatmap

# In[22]:


importances_data = []
for shape_type, df in dataframes_updated.items():
    for target_var in ['Strict', 'Relaxed']:
        X = df[['dist', 'angle', 'clustd']]
        y = df[target_var]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_

        importances_data.append({
            'Shape Type': f"{shape_type}_{target_var}",
            'Distance': importances[0],
            'Angle': importances[1],
            'Standard Deviation': importances[2]
        })

importances_df = pd.DataFrame(importances_data)
importances_df = importances_df.set_index('Shape Type')

plt.figure(figsize=(12, 10))
sns.heatmap(importances_df, annot=True, cmap='YlGnBu', fmt='.3f', annot_kws={"size": 18} ,cbar_kws={'label': 'Information Gain'}) #YlGnBu
plt.xlabel('Outlier parameters varied')
plt.ylabel('Inlier shapes and ground truth categories')
plt.title('Feature Importance', weight='bold')
plt.tight_layout()
plt.savefig(f'{output_folder_path}/heatmap_OSTI_Case1.png',dpi=600)
plt.close()


# ## Time compilation

# Function to extract time values from all the required subfolders
def extract_time_values(file_content):
    wall_clock_time = re.search(r'The script took ([\d.]+) seconds \(wall clock time\) to run\.', file_content)
    cpu_time = re.search(r'The script took ([\d.]+) seconds of CPU time to run\.', file_content)
    memory_used = re.search(r'The script used (\d+) bytes of memory\.', file_content)

    wall_clock_seconds = float(wall_clock_time.group(1)) if wall_clock_time else 0
    cpu_seconds = float(cpu_time.group(1)) if cpu_time else 0
    memory_bytes = int(memory_used.group(1)) if memory_used else 0
    
    return wall_clock_seconds, cpu_seconds, memory_bytes

# Base directory containing the subfolders
base_folder = '../results/Case1_datasets'
required_subfolders = {'circle', 'ellipse', 'triangle', 'no_transform'}  # Ensure all the results for each of the shape files is there.

# Open a file to write the output
with open(f'{output_folder_path}/time_metrics_case1.txt', 'w') as f:
    # Iterate through each required subfolder
    for subfolder in required_subfolders:
        directory = os.path.join(base_folder, subfolder, f'results_{subfolder}')
        
        total_wall_clock_time = 0
        total_cpu_time = 0
        
        # Iterate through all files in the subfolder
        for filename in os.listdir(directory):
            if filename.startswith('df') and filename.endswith('results.txt'):
                with open(os.path.join(directory, filename), 'r') as file:
                    content = file.read()
                    wall_clock_seconds, cpu_seconds, memory_bytes = extract_time_values(content)
                    total_wall_clock_time += wall_clock_seconds
                    total_cpu_time += cpu_seconds

        # Convert total time from seconds to minutes
        total_wall_clock_time_minutes = total_wall_clock_time / 60
        total_cpu_time_minutes = total_cpu_time / 60

        # Write the results to the file
        f.write(f"Subfolder: {subfolder}\n")
        f.write(f"  Total wall clock time in minutes: {total_wall_clock_time_minutes:.2f}\n")
        f.write(f"  Total CPU time in minutes: {total_cpu_time_minutes:.2f}\n")






