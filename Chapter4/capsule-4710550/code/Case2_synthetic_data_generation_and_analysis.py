#!/usr/bin/env python
# coding: utf-8

# ## This routine is for generating synthetic datasets for case 2 and analsyisng them. All the figures added in paper titled "A Robust Two-step Method for Clustering and Dealing with Outlier Sets in Big Data". For questions contact asarfraz1@sheffield.ac.uk 

# ## Importing libraries

# In[1]:


import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
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


# In[2]:


from case2 import save_dataset,no_transform,circle_transform,ellipse_transform,triangle_transform,extract_values_from_filename,plot_case2,gt_strict,gt_relaxed,overlap_between_outliers,cluster_mahala,compute_classification_metrics,compute_cluster_purity,compute_cluster_purity_both_out,get_purities_both,plot_purities_updated,classify_outliers,classify


# In[3]:


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


# ## Synthetic Dataset Generation for Case 2 (Two Outlier Set)

# In[4]:


# Generate and save datasets
transform_funcs = {
    "circle": (circle_transform, (5, 5)),
    "ellipse": (ellipse_transform, (5, 10)),
    "triangle": (triangle_transform, (-20, 20, -20, 20,6)),
    "no_transform": (no_transform, ())
}

# Set total datasets and distributions per shape
n_datasets = 4000
n_per_shape = 1000 # Datasets per shape
n_datapoints = 1500
n_outliers = 150

# Define parameter ranges for both sets of clusters
param_ranges = {
    'cluster_std_range': [1, 10],
    'desired_distance_range': [0, 100],
    'desired_angle_range': [0, 360],
    'cluster_std_range2': [1, 10],
    'desired_distance_range2': [0, 100],
    'desired_angle_range2': [0, 360]
}

# Generate samples using LHS
samples = lhs(6, samples=n_datasets)
params = {key: val[0] + samples[:, idx] * (val[1] - val[0]) for idx, (key, val) in enumerate(param_ranges.items())}

shape_transforms = ['circle', 'ellipse', 'triangle', 'no_transform']



# In[5]:


# Create and save datasets
for i in range(n_datasets):
    cluster_std1 = params['cluster_std_range'][i]
    desired_distance1 = params['desired_distance_range'][i]
    desired_angle1 = params['desired_angle_range'][i]
    cluster_std2 = params['cluster_std_range2'][i]
    desired_distance2 = params['desired_distance_range2'][i]
    desired_angle2 = params['desired_angle_range2'][i]

    X_datapoints, _ = make_blobs(n_samples=n_datapoints, centers=15, cluster_std=6, center_box=(-10, 10))
    inliers = X_datapoints.copy()
    X_outliers1, _ = make_blobs(n_samples=n_outliers // 2, centers=1, cluster_std=cluster_std1)
    X_outliers2, _ = make_blobs(n_samples=n_outliers // 2, centers=1, cluster_std=cluster_std2)

    angle_rad1 = np.deg2rad(desired_angle1)
    outliers_centroid1 = inliers.mean(axis=0) + np.array([desired_distance1 * np.cos(angle_rad1), desired_distance1 * np.sin(angle_rad1)])
    X_outliers1 += outliers_centroid1 - X_outliers1.mean(axis=0)

    angle_rad2 = np.deg2rad(desired_angle2) +np.pi + angle_rad1 #angle_rad_opposite = angle_rad + np.pi + np.deg2rad(desired_angle2)
    outliers_centroid2 = inliers.mean(axis=0) + np.array([desired_distance2 * np.cos(angle_rad2), desired_distance2 * np.sin(angle_rad2)])
    X_outliers2 += outliers_centroid2 - X_outliers2.mean(axis=0)

    dataset = np.vstack([inliers, X_outliers1, X_outliers2])
    is_outlier = np.vstack([np.zeros((n_datapoints, 1), dtype=bool), np.ones((n_outliers // 2, 1), dtype=bool), np.ones((n_outliers // 2, 1), dtype=bool)])
    labels = np.vstack([np.zeros((n_datapoints, 1), dtype=int), np.ones((n_outliers // 2, 1), dtype=int), np.full((n_outliers // 2, 1), 2, dtype=int)])

    shape_index = i // n_per_shape
    transform_name = shape_transforms[shape_index]
    transformed_inliers = transform_funcs[transform_name][0](inliers, *transform_funcs[transform_name][1])
    transformed_data = np.vstack([transformed_inliers, X_outliers1, X_outliers2])

    final_dataset = np.hstack([transformed_data, is_outlier, labels])
    df = pd.DataFrame(final_dataset, columns=["Feature 1", "Feature 2", "is_outlier", "outlier_set"])
#=======================================give folder path here=========================
    folder_path = f"../results/Case2_datasets/{transform_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
#=================================================================================
    filename = f"df_{i}_std1_{round(cluster_std1, 2)}_dist1_{round(desired_distance1, 2)}_ang1_{round(desired_angle1, 2)}_std2_{round(cluster_std2, 2)}_dist2_{round(desired_distance2, 2)}_ang2_{round(desired_angle2, 2)}.pkl"
    df.to_pickle(os.path.join(folder_path, filename))
#=============additional check be implemented to recheck everything    
    # Save outlier centroids to a .txt file
    #centroid_filename = f"centroid_{i}_std1_{round(cluster_std1, 2)}_dist1_{round(desired_distance1, 2)}_ang1_{round(desired_angle1, 2)}_std2_{round(cluster_std2, 2)}_dist2_{round(desired_distance2, 2)}_ang2_{round(desired_angle2, 2)}.txt"
    #centroid_filepath = os.path.join(folder_path, centroid_filename)
    
    #with open(centroid_filepath, 'w') as file:
    #    file.write(f"Outlier Set 1 Centroid: {outliers_centroid1}\n")
    #    file.write(f"Outlier Set 2 Centroid: {outliers_centroid2}\n")

#print("All Case 2 datasets were generated successfully.")


# *Plotting of the datasets is optional and would required 35-45 minutes*

# In[6]:


# #============optional plotting of datasets generated Case 2
# for transform_name in ["circle", "ellipse", "triangle", "no_transform"]:
#     #========================================Ensure this matches your saving directory in the previous step==========================
#     transform_folder = os.path.join('Case2_datasets', transform_name)  
#     for filename in os.listdir(transform_folder):
#         if filename.endswith(".pkl"):
#             # Load the dataset
#             df = pd.read_pickle(os.path.join(transform_folder, filename))
            
#             # Plot the dataset
#             plot_filename = filename.replace('.pkl', '.png')
#             plot_folder = os.path.join(transform_folder, 'plots')
#             if not os.path.exists(plot_folder):
#                 os.makedirs(plot_folder)
#             plot_case2(df, plot_folder, plot_filename)

# print("All Case 2 datasets plotted and saved successfully.")


# ## Running OSTI for Case 2

# *The analysis below takes 40 minutes on average to run for each requried_subfolders, based on your hardware specififcations please run for each folder iteratievly or on all folders automatically*

# In[7]:


base_folder = '../results/Case2_datasets'


# In[8]:


#for running one folder at a time uncomment the below lines for each shape (40 minutes on average for each folder)
# Ensure to have all the shapes results run before heading to the analysis bit 
#required_subfolders = {'circle'}
#required_subfolders = {'ellipse'}
#required_subfolders = {'triangle'}
#required_subfolders = {'no_transform'}


# In[9]:


#for running all folders (will take approx 2-3 hours)
required_subfolders = {'circle','ellipse','triangle', 'no_transform'}


# In[10]:


for shape in required_subfolders:
    shape_path = os.path.join(base_folder, shape)
    if not os.path.exists(shape_path):
        print(f"Shape subfolder not found: {shape_path}")
        continue 

    pkl_files = glob.glob(os.path.join(shape_path, '*.pkl'))
    OSTI_results = []
    gt_strict_results = []
    gt_relaxed_results = []
    gts_overlap = []
    gtr_overlap = []
    outlier_labels_list = []
#============================================================
    n_clusters = 8
    palette = sns.color_palette('colorblind', n_clusters)
#============================================================
    

    start_wall_time = time.time()
    start_cpu_time = time.process_time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    
    for pkl_file in pkl_files:
        data = pd.read_pickle(pkl_file)
        transform_name = shape
        filename = os.path.basename(pkl_file).replace('.pkl', '')
#================================================================================================================================== 
        results_subfolder = os.path.join(shape_path, f'results_{shape}')
        os.makedirs(results_subfolder, exist_ok=True)
#================================================================================================================================== 
        X = data[['Feature 1', 'Feature 2']].copy()
        outlier_labels = data['is_outlier']
        outliers = data['outlier_set'].astype(int)
        
        outliers_set_1 = data['outlier_set'] == 1
        outliers_set_2 = data['outlier_set'] == 2
        outlier_set_labels = data['outlier_set']
        #outlier_set_labels = np.concatenate((np.full(n_datapoints, 0), np.full(n_outliers, 1),np.full(n_outliers, 2))) 
               

        XX = X.iloc[:,[0,1]]

#================================================================================================================                
        start_wall_time1 = time.time()
        start_cpu_time1 = time.process_time()
        process1 = psutil.Process(os.getpid())
        start_memory1 = process.memory_info().rss
           
        X, XX_values,cluster_labels, cluster_weights, cluster_covariances, cluster_means, cluster_stats_df, clusters_df, classification_metrics, normalised_distances, p_values=cluster_mahala(XX, X, n_clusters, outlier_labels, compute_classification_metrics,random_seed=42, weight_thres=0.1, alpha_thres=0.05)


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
#========================================================#========================================================                
        df_labeled = pd.concat([data,clusters_df], axis=1)
        X=df_labeled
        #calculating via multiple methods to check 
        cluster_purity=compute_cluster_purity(n_clusters, outlier_labels, cluster_labels)
        
        cluster_purity1,cluster_purity2=compute_cluster_purity_both_out(n_clusters, outlier_labels, cluster_labels, outlier_set_labels)
        
        #=======================================================plotting===========================================
        fig, axes = plt.subplots(1, 3, figsize=(30,10)) 
        #=================================================================first plot=================================================================
        #outliers = X[X['is_outlier']]
        outliers1 = outliers_set_1
        outliers2 = outliers_set_2
        datapoints=X[X['is_outlier'] == 0]
        axes[0].scatter(datapoints.iloc[:, 0], datapoints.iloc[:, 1], s=60, alpha=0.6, label='datapoints', marker='o')
        axes[0].scatter(X.loc[outliers_set_1, 'Feature 1'], X.loc[outliers_set_1, 'Feature 2'], s=60, alpha=0.9, label='Outliers1', marker='^', color='red')
        axes[0].scatter(X.loc[outliers_set_2, 'Feature 1'], X.loc[outliers_set_2, 'Feature 2'], s=60, alpha=0.9, label='Outliers2', marker='s', color='black')
        axes[0].set_title(f'Clustering Result', weight='bold')
        axes[0].set_xlabel('Feature 1', weight='bold')
        axes[0].set_ylabel('Feature 2', weight='bold')
        axes[0].legend(bbox_to_anchor=(1.2, 1),loc='upper right')
        #=================================================================Second plot=================================================================
        XX_values = XX.values
        #axes[1].scatter(outliers1.iloc[:, 0], outliers1.iloc[:, 1], s=100, alpha=0.9, label='Outliers1', marker='^')
        #axes[1].scatter(outliers2.iloc[:, 0], outliers2.iloc[:, 1], s=100, alpha=0.9, label='Outliers2', marker='s')
        axes[1].scatter(X.loc[outliers_set_1, 'Feature 1'], X.loc[outliers_set_1, 'Feature 2'], s=60, alpha=0.9, label='Outliers1', marker='^', color='red')
        axes[1].scatter(X.loc[outliers_set_2, 'Feature 1'], X.loc[outliers_set_2, 'Feature 2'], s=60, alpha=0.9, label='Outliers2', marker='s', color='black')
        
        for i in range(n_clusters):
            cluster_points = XX_values[cluster_labels == i]
            color = palette[i]
            axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1],s=50, color=color, alpha=0.7, label='Cluster {} Weight={:.2f}'.format(i, cluster_weights[i]))
            axes[1].text(cluster_means[i, 0], cluster_means[i, 1], str(i), color=color, ha='center', va='center', fontsize=12, weight='bold', bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.5'))# Added this line
            if cluster_weights[i] <= 0.1:
                for point in cluster_points:
                    axes[1].plot(point[0], point[1], marker='+', markersize=5, color="black")
        axes[1].set_title('Candidate OSTI', weight='bold')
        axes[1].set_xlabel('Feature 1', weight='bold')
        axes[1].set_ylabel('Feature 2', weight='bold')
        axes[1].legend(bbox_to_anchor=(1.2, 1.2), title="Clusters and their weights", loc='upper right')
        #=================================================================Third plot=================================================================

        
        highlight_green = [True if w <= 0.1 and p <= 0.05 else False for w, p in zip(cluster_weights, p_values)]
        colors = ['green' if h_green else 'blue' for h_green in highlight_green]
        sizes = [400 if h_green else 250 for h_green in highlight_green]
        padding = 0.05
        
        axes[2].scatter(normalised_distances, p_values, c=colors, s=sizes)
        legend_elements_purity = []

        # Include purity information in the legend or directly on the plot

        legend_elements_purity = []

        for i in range(n_clusters):
            color = 'green' if highlight_green[i] else 'blue'
            #label_text = f'Cluster {i}, Purity Set 1: {cluster_purity_set1[i]:.2f}, Purity Set 2: {cluster_purity_set2[i]:.2f}'
            label_text = f'Cluster {i}, Purity: {cluster_purity[i]:.2f}'
            legend_elements_purity.append(plt.Line2D([0], [0], marker='o', color='w', label=label_text, markerfacecolor='black', markersize=10))
        axes[2].legend(handles=legend_elements_purity, loc='upper right', title="Cluster Purity",bbox_to_anchor=(0.99, 1.0))
        axes[2].text(normalised_distances[i], p_values[i], 
             f'{i}\nP1: {cluster_purity[i]:.2f}', 
             fontsize=10, ha='center', va='center', color='white')
        axes[2].set_xlabel('Normalised Mahalanobis Distance', weight='bold')
        axes[2].set_ylabel('P-value', weight='bold')
        plt.yticks(np.arange(0, np.ceil(max(p_values))+0.1, step=0.1))
        plt.ylim(0 - padding, np.ceil(max(p_values)) + padding)
        axes[2].set_title('OSTI with cluster purity', weight='bold')
        for i,(normalised_distance,p_value) in enumerate(zip(normalised_distances, p_values)):
            axes[2].text(normalised_distance,p_value,f'{i}', fontsize=12, ha='center', va='center', color='white')
        axes[2].grid(True)
        fig.tight_layout()

        fig.savefig(os.path.join(results_subfolder, f"{filename}.png"), dpi=30, bbox_inches='tight')
        plt.close()

        has_green_cluster = any(color == 'green' for color in colors)
        cluster_info = clusters_df.to_dict('records')
        green_clusters = [i for i, is_green in enumerate(highlight_green) if is_green]


        X_datapoints = X.loc[X['is_outlier'] == 0, ['Feature 1', 'Feature 2']]
        datapoints_centroid = X_datapoints.mean().values
        
        outliers_centroid1 = X[outliers_set_1 & X['is_outlier']].iloc[:, :2].mean().values
        outliers_centroid2 = X[outliers_set_2 & X['is_outlier']].iloc[:, :2].mean().values
        
        # Extract outliers set 1
        X_outliers1 = X[outliers_set_1 & X['is_outlier']].iloc[:, :2].values
        
        # Extract outliers set 2
        X_outliers2 = X[outliers_set_2 & X['is_outlier']].iloc[:, :2].values
        gts_outliers1, gts_outliers2= gt_strict(X_datapoints, X_outliers1, X_outliers2)
        gt_strict_results.append((gts_outliers1, gts_outliers2))
        
        gtr_outliers1, gtr_outliers2= gt_relaxed(X_datapoints, outliers_centroid1, outliers_centroid2)
        gt_relaxed_results.append((gtr_outliers1, gtr_outliers2))

        overlap = overlap_between_outliers(X_outliers1, X_outliers2)

        # Save centroids to a text file ===== additional check
        #centroid_filename = f"{filename}_centroids.txt"
        #centroid_filepath = os.path.join(results_subfolder, centroid_filename)
        
        #with open(centroid_filepath, 'w') as f:
        #    f.write(f"Datapoints Centroid: {datapoints_centroid}\n")
        #    f.write(f"Outliers Centroid 1: {outliers_centroid1}\n")
        #    f.write(f"Outliers Centroid 2: {outliers_centroid2}\n")
                
        # Convert 'cluster_labels' column to integer type
        X['cluster_labels'] = X['cluster_labels'].astype(int)

        green_has_outlier1 = any(X[(X['cluster_labels'] == green_cluster) & (X['outlier_set'] == 1)]['is_outlier'].any() for green_cluster in green_clusters)
        green_has_outlier2 = any(X[(X['cluster_labels'] == green_cluster) & (X['outlier_set'] == 2)]['is_outlier'].any() for green_cluster in green_clusters)
        
        set_num, std1, dist1, ang1, std2, dist2, ang2 = extract_values_from_filename(filename)
        
        OSTI_results.append({'filename': filename, 
                             'OSTI_identified': 'yes' if has_green_cluster else 'no', 
                             'OS_1': 'yes' if green_has_outlier1 else 'no',
                             'OS_2': 'yes' if green_has_outlier2 else 'no', 
                             'OSTI_both': 'yes' if green_has_outlier1 and green_has_outlier2 else 'no',
                             'gts_outliers1': gts_outliers1,
                             'gts_outliers2': gts_outliers2,
                             'gtr_outliers1': gtr_outliers1,
                             'gtr_outliers2': gtr_outliers2,
                             'set_num': set_num,
                             'std1': std1,
                             'dist1': dist1,
                             'ang1': ang1,
                             'std2': std2,
                             'dist2': dist2,
                             'ang2': ang2,
                             'overlap_O1_O2': overlap,
                             'cluster_info': cluster_info,
                             'p_values': p_values,
                             'cluster_purity': cluster_purity,
                             'cluster_purity1': cluster_purity1,
                             'cluster_purity2': cluster_purity2,
                             'outlier_labels': outlier_set_labels.tolist(),
                             'cluster_labels': cluster_labels.tolist(),
                             'cluster_info': cluster_info,
                             **classification_metrics})
#=================================================================RESULTS FOLDER NAME=================================================================        
        OSTI_analysis = pd.DataFrame(OSTI_results)
        OSTI_analysis.to_pickle(os.path.join(results_subfolder, f'{transform_name}_OS2.pkl'))
#================================================================================================================================== 

    # Example placeholder for further steps (replace with actual computation and plotting)
        #print(f"Running analysis for {transform_name} transformation")
    
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

#print("All analysis on Case 2 datasets was conducted successfully.")


# ## Analysing results

# In[11]:


# Saving all the compiled results for all shapes to a new folder for further analysis 
base_folder = '../results/Case2_datasets'
required_subfolders = {'circle', 'ellipse', 'triangle', 'no_transform'} #for the analysis bit please ensure you have all the results for each shape 
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


# In[12]:


#========================Update path accordingly==============================
folder_path = new_analysis_folder

output_folder_path = '../results/Case2_datasets/Case2_FINAL_result_pickles'
#========================Update path accordingly==============================


if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# List all pickle files in the folder
pickle_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

# Process each pickle file
for file in pickle_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_pickle(file_path)
    
    ground_truths = ['gts_outliers1', 'gtr_outliers1', 'gts_outliers2', 'gtr_outliers2']
    
    for ground_truth in ground_truths:
        classification_column = f'classification_{ground_truth}'
        df[classification_column] = df.apply(lambda row: classify(row, ground_truth), axis=1)
    
    df['012_mapping'] = df.apply(classify_outliers, axis=1)
    
    df[['outlier_cluster_purity_set1', 'outlier_cluster_purity_set2']] = df.apply(get_purities_both, axis=1, result_type='expand')
    
    # Calculate the sum of purities for each outlier set
    df['outlier_1_purity_sum'] = df['outlier_cluster_purity_set1'].apply(lambda x: sum(x))
    df['outlier_2_purity_sum'] = df['outlier_cluster_purity_set2'].apply(lambda x: sum(x))
    
    # Format the purity sum values to 2 decimal places
    df['outlier_1_purity_sum'] = df['outlier_1_purity_sum'].apply(lambda x: '{:.2f}'.format(x))
    df['outlier_2_purity_sum'] = df['outlier_2_purity_sum'].apply(lambda x: '{:.2f}'.format(x))
    
    base_filename = os.path.splitext(file)[0]
    modified_filename = f"{base_filename}_results.pkl"
    output_file_path = os.path.join(output_folder_path, modified_filename)
    
    df.to_pickle(output_file_path)

#print("Processing complete.")


# In[13]:


circle_filename = 'circle_OS2_results.pkl'
ellipse_filename = 'ellipse_OS2_results.pkl'
triangle_filename = 'triangle_OS2_results.pkl'
no_transform_filename = 'no_transform_OS2_results.pkl'

df_os2_c = pd.read_pickle(os.path.join(output_folder_path, circle_filename))
df_os2_e = pd.read_pickle(os.path.join(output_folder_path, ellipse_filename))
df_os2_t = pd.read_pickle(os.path.join(output_folder_path, triangle_filename))
df_os2_i = pd.read_pickle(os.path.join(output_folder_path, no_transform_filename))


# In[14]:


dataframes_updated = {
    'Circle': df_os2_c,
    'Ellipse': df_os2_e,
    'Triangle': df_os2_t,
    'Irregular': df_os2_i
}


# ## Purity calculations and plotting 

# In[15]:


# #========= additional check for purity calculations ========
# column_to_check = ['outlier_1_purity_sum', 'outlier_2_purity_sum']  # Replace with the desired column names
# greater_than_one = True

# for file in os.listdir(output_folder_path):
#     if file.endswith('_results.pkl'):
#         file_path = os.path.join(output_folder_path, file)
#         df = pd.read_pickle(file_path)
#         if not ((df[column_to_check].astype(float) > 1).any(axis=1)).all():
#             greater_than_one = False
#             break

# if greater_than_one:
#     print(f"All updated dataframes have a value greater than one in columns {column_to_check}")
# else:
#     print(f"Not all updated dataframes have a value greater than one in columns {column_to_check}")


# ## updated purity plotting code below

# In[16]:


plot_purities_updated(dataframes_updated, output_folder_path)


# In[ ]:





# ## Classification of GT count,F1-score and plots

# In[17]:


classification_columns = [
    'classification_gts_outliers1', 'classification_gts_outliers2',
    'classification_gtr_outliers1', 'classification_gtr_outliers2'
]

# Initialising dictionaries to hold the sum of TP, FP, FN, and TN for strict and relaxed classifications
sums_strict = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
sums_relaxed = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

# Open a file to write the output
with open(f'{output_folder_path}/classification_metrics_case2.txt', 'w') as f:
    # Iterate over the dataframes and aggregate the counts
    for shape, df in dataframes_updated.items():
        f.write(f"Processing {shape} shape...\n")
        for column in classification_columns:
            value_counts = df[column].value_counts()

            TP = value_counts.get('TP', 0)
            FP = value_counts.get('FP', 0)
            FN = value_counts.get('FN', 0)
            TN = value_counts.get('TN', 0)

            # Sum the counts for strict and relaxed classifications
            if 'gts' in column:
                sums_strict['TP'] += TP
                sums_strict['FP'] += FP
                sums_strict['FN'] += FN
                sums_strict['TN'] += TN
            elif 'gtr' in column:
                sums_relaxed['TP'] += TP
                sums_relaxed['FP'] += FP
                sums_relaxed['FN'] += FN
                sums_relaxed['TN'] += TN

            # Write the value counts for the current column to the file
            f.write('----------------------\n')

    # Calculate precision, recall, and F1 score for strict classification
    precision_strict = sums_strict['TP'] / (sums_strict['TP'] + sums_strict['FP']) if (sums_strict['TP'] + sums_strict['FP']) != 0 else 0
    recall_strict = sums_strict['TP'] / (sums_strict['TP'] + sums_strict['FN']) if (sums_strict['TP'] + sums_strict['FN']) != 0 else 0
    f1_strict = 2 * (precision_strict * recall_strict) / (precision_strict + recall_strict) if (precision_strict + recall_strict) != 0 else 0
    
    # Calculate precision, recall, and F1 score for relaxed classification
    precision_relaxed = sums_relaxed['TP'] / (sums_relaxed['TP'] + sums_relaxed['FP']) if (sums_relaxed['TP'] + sums_relaxed['FP']) != 0 else 0
    recall_relaxed = sums_relaxed['TP'] / (sums_relaxed['TP'] + sums_relaxed['FN']) if (sums_relaxed['TP'] + sums_relaxed['FN']) != 0 else 0
    f1_relaxed = 2 * (precision_relaxed * recall_relaxed) / (precision_relaxed + recall_relaxed) if (precision_relaxed + recall_relaxed) != 0 else 0
    
    # Write the final precision, recall, and F1 scores to the file
    f.write('Strict Classification:\n')
    f.write(f'Precision: {precision_strict}\n')
    f.write(f'Recall: {recall_strict}\n')
    f.write(f'F1 Score: {f1_strict}\n')
    f.write('=========================\n')
    
    f.write('Relaxed Classification:\n')
    f.write(f'Precision: {precision_relaxed}\n')
    f.write(f'Recall: {recall_relaxed}\n')
    f.write(f'F1 Score: {f1_relaxed}\n')
    f.write('=========================\n')


# In[18]:


#===============================This optional plotting is only for checking ====================
# classifications = ['classification_gts_outliers1', 'classification_gts_outliers2', 'classification_gtr_outliers1', 'classification_gtr_outliers2']
# color_mapping = {'TP': 'green', 'TN': '#56B4E9', 'FP': 'black', 'FN': 'yellow'}

# for shape, df in dataframes_updated.items():
#     print(f"Processing {shape} shape...")
#     filtered_df = df[df['overlap_O1_O2'] == True]

#     for classification in classifications:
#         fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
#         # Determine whether to plot for outliers1 or outliers2
#         if 'outliers1' in classification:
#             plots = [('dist1', 'ang1'), ('dist1', 'std1'), ('std1', 'ang1')]
#         elif 'outliers2' in classification:
#             plots = [('dist2', 'ang2'), ('dist2', 'std2'), ('std2', 'ang2')]
        
#         # Iterate through subplots
#         for i, (xlabel, ylabel) in enumerate(plots):
#             ax = axes[i]
#             sns.scatterplot(data=df, x=xlabel, y=ylabel, hue=classification, ax=ax, legend=False, palette=color_mapping)
#             #sns.scatterplot(data=filtered_df, x=xlabel, y=ylabel, ax=ax, facecolors='none', edgecolor='black', linewidth=1.2, legend=False)
#             sns.scatterplot(data=filtered_df, x=xlabel, y=ylabel, ax=ax, color='red', marker='^',s=80,legend=False)

#             ax.set_xlim([0, 100] if 'dist' in xlabel else [1, 10])
#             ax.set_ylim([0, 360] if 'ang' in ylabel else [1, 10])
#             ax.set_xlabel(xlabel.capitalize())
#             ax.set_ylabel(ylabel.capitalize())
        
#         # Create custom patches for the legend
#         class_counts = df[classification].value_counts().to_dict()
#         patches = [
#             mpatches.Patch(color=color, label=f'{class_label}: {class_counts[class_label]}')
#             for class_label, color in color_mapping.items() if class_label in class_counts
#         ]
#         fig.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#         plt.suptitle(f"{classification} - {shape}")
#         plt.tight_layout()
#         plt.show()
#==========================================================================================================================


# In[19]:


# Mapping for 0, 1, 2 outliers zone
colors_012 = {
    0: 'black',  # No OS
    1: '#56B4E9',  # One OS
    2: 'green'  # Both OS
}

# Mapping for detection status
detection_status = {
    0: "No OS",  # black
    1: "1 OS",  # bluish green
    2: "2 OS"  # green
}

# Define the desired order of dataframes
plot_order = ["Circle", "Ellipse", "Triangle", "Irregular"]

# Create custom legend patches including the yellow star for overlap
legend_patches = [mpatches.Patch(color=colors_012[k], label=detection_status[k]) for k in colors_012.keys()]
legend_patches.append(mlines.Line2D([], [], color='yellow', marker='*', linestyle='None', markersize=10, label='Overlap Region'))

fig, axes = plt.subplots(3, 4, figsize=(24, 18))

for row, plot_type in enumerate(['dist', 'ang', 'std']):
    for col, key in enumerate(plot_order):
        df = dataframes_updated[key]
        filtered_df = df[df['overlap_O1_O2'] == True]
        
        if plot_type == 'dist':
            x_label, y_label = 'dist1', 'dist2'
            x_lim, y_lim = (0, 100), (0, 100)
        elif plot_type == 'ang':
            x_label, y_label = 'ang1', 'ang2'
            x_lim, y_lim = (0, 360), (0, 360)
        else:
            x_label, y_label = 'std1', 'std2'
            x_lim, y_lim = (1, 10), (1, 10)
        
        axes[row, col].scatter(df[x_label], df[y_label], c=df['012_mapping'].map(colors_012))
        axes[row, col].scatter(filtered_df[x_label], filtered_df[y_label], color='yellow', marker='*', s=30)
        axes[row, col].set_xlabel(x_label)
        axes[row, col].set_ylabel(y_label)
        axes[row, col].set_xlim(x_lim)
        axes[row, col].set_ylim(y_lim)
        axes[row, col].set_title(key)
    
plt.tight_layout(rect=[0, 0, 0.95, 0.95])

# Add a single legend for all plots
fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(0.65, -0.05),ncol=2)
plt.savefig(f'{output_folder_path}/case2_param1vs2.png', dpi=600, bbox_inches='tight')
plt.close()


# ## Heatmap Case 2

# In[20]:


importances_data = []

for shape_type, df in dataframes_updated.items():
    for target_var in ['classification_gts_outliers1', 'classification_gtr_outliers1', 'classification_gts_outliers2', 'classification_gtr_outliers2']:
        X = df[['dist1', 'ang1', 'std1', 'dist2', 'ang2', 'std2']]
        y = df[target_var]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_

        # Update the target_var name for better readability
        target_var_readable = (target_var
                               .replace('classification_gtr', 'Relaxed')
                               .replace('classification_gts', 'Strict')
                               .replace('outliers1', 'OS1')
                               .replace('outliers2', 'OS2'))
        
        importances_data.append({
            'Shape Type': f"{shape_type}_{target_var_readable}",
            'Distance1': importances[0],
            'Angle1': importances[1],
            'Standard Deviation1': importances[2],
            'Distance2': importances[3],
            'Angle2': importances[4],
            'Standard Deviation2': importances[5]
        })

importances_df = pd.DataFrame(importances_data)
importances_df = importances_df.set_index('Shape Type')

plt.figure(figsize=(18, 10))
heatmap = sns.heatmap(importances_df, annot=True, cmap='YlGnBu', fmt='.3f', annot_kws={"size": 18}, cbar_kws={'label': 'Information Gain'})  # YlGnBu
plt.xlabel('Outlier parameters varied')
plt.ylabel('Inlier shapes and ground truth categories for each OS')
plt.title('Feature Importance', weight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_folder_path}/heatmap_OSTI_Case2.png', dpi=600)
plt.close()


# ## Time compilation

# In[21]:


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
base_folder = '../results/Case2_datasets'
required_subfolders = {'circle', 'ellipse', 'triangle', 'no_transform'}  # Ensure all the results pickle files are generated.

# Open a file to write the output
with open(f'{output_folder_path}/time_metrics_case2.txt', 'w') as f:
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







