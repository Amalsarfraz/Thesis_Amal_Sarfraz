import subprocess

notebooks = ["Figure_1_IRB_all_methods.py", "Case2_synthetic_data_generation_and_analysis.py","Case1_synthetic_data_generation_and_analysis.py"]

for notebook in notebooks:
    subprocess.run(["python", notebook], check=True)