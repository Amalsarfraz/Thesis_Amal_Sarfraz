# OSTI_OutlierSet_Twostep_Identification

 

**Setup**

1. To create a conda environment with all the dependencies, we create an environment.yml file with the neccessary libraries required (the file is already in the folder).

From anaconda prompt go to destination folder and give the following command 

conda env create -f environment.yml

Now activate the environment along with jupyter notebook
 
conda activate osti

After activating the environment, you can start run python main.py from anaconda prompt.


**Usage**

2. The python scripts `main.py` has all the validation runs. Please note this script will take time as it runs and analyzes 8000 synthetic datasets.

4. The python scripts `case1.py` and `case2.py` have all the functions defined. 

5. Run python scripts `Case1_synthetic_data_generation_and_analysis.py` and `Case2_synthetic_data_generation_and_analysis.py` for case 1 and case 2 respectively to generate synthetic datasets and analyse them. All the figures in paper and appendix are added in these two notebooks (scatter plots, box plots and heatmaps).

6. The script `Figure_1_IRB_all_methods.py` runs and generates figure 1 from the paper.

**Data files**

7. `2100_cotton.csv`: A subset of data from Dolan, F., Lamontagne, J., Link, R. et al. 'Evaluating the economic impact of water scarcity in a changing world'. Nat Commun 12, 1915 (2021). https://doi.org/10.1038/s41467-021-22194-0."

8. These routines are supplement to the paper titled `A Robust Two-step Method for Clustering and Dealing with Outlier Sets in Big Data`. 

For any further queries please contact asarfraz1@sheffield.ac.uk 
