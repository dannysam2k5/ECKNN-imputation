######################################################################
# Combined local neighbors and clustering-wise weights for KNNimpute
######################################################################

This project seeks to implement a novel technique called Enhanced Clustering-based K-Nearest Neighbor (ECKNN) imputation  which combines sample-level local neighbors with cluster-wise global weights to improve imputation accuracy.

## Notebooks

- ECKNNimpute_das108_workbook.ipynb	: This is the original notebook that offers a thorough explanation of the experiment that was carried out, along with a step-by-step analysis and all the implemented functions.

- ECKNNimpute_das108_Simplified_workbook.ipynb : This is a simplified version of the original notebook that only focuses on the experimental analysis of 3 different imputation methods - ECKNN, feature average (mean imputation), and KNN imputation. The notebook imports a module named imputation_utils.py which contains all the functions from the original notebook.

## Analysis Overview

- ECKNNimpute_das108_workbook.ipynb

This notebook contains the following steps:

	1. Importing Python Packages
    2. Reading Datasets
    3. Data Exploration and Data pre-processing: Data Cleaning & Feature Engineering (if needed)
    4. Data Normalization
    5. Perform K-means Clustering and Combined Weight Matrix
    6. Perform artificial missingness with missing Rates
		6.1 Validation of missing rate
    7. Imputation process
		7.1 Perform initial imputation using feature average
		7.2 Perform advanced imputation using pairwise similarity matrix
    8. Performance Evaluation Metrics
    9. Experimental Design
		9.1 Experimental datasets
		9.2 Experimental run
		9.3 Visualisation (bar charts + line charts)

- ECKNNimpute_das108_Simplified_workbook.ipynb

This notebook contains the following steps:
	
	1. Experimental Design
		1.1 Experimental datasets
		1.2 Experimental run
		1.3 Visualisation (bar charts + line charts)
		
The original notebook contains all the code implemented directly in the notebook. The simplified notebook imports and applies reusable functions from the module imputation_utils.py

## Usage

	1. ECKNNimpute_das108_workbook.ipynb:
		- Use Jupyter or any compatible platform to access the notebook.
		- Execute the cells in a top-down approach starting from the first cell.
		
	2. ECKNNimpute_das108_Simplified_workbook.ipynb
		- Ensure the module imputation_utils.py is made available in the same directory of your notebook environment.
		- Execute the cells in a top-down approach starting from the first cell.
		
## Data

The experiment makes use of dataset, which must be made available.

## Author
	
	Daniel Sam-Egbede (das108@aber.ac.uk / dannysam2k5@gmail.com)
	
	
## Acknowledgments

I would like to acknowledge and express my gratitude to Prof. Tossapon Boongoen who provided guidance on this experiment.
