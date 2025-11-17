# MoDAmix
A Unified Framework for Correcting Batch Effects and Integrating Multi-Omics Data

## Requirements
* Python (>= 3.6)
* Pytorch (>= v1.6.0)
* Other python packages : numpy (>=1.19.1), pandas (>=1.1.1), os, sys, random

## Usage
Clone the repository or download source code files.

## Inputs
[Note!] All the example datasets can be found in './example_data/' directory.

### 1. Source multi-omics dataset
* First omics profiles to be used as source domain (**Source_omics1_X, Source_omics2_X**)
  - Row : Sample, Column : Feature
  - The first column should have "sample id", and the first row should contain the feature names
  - Samples should be sorted in the same order for both omics 1 and omics 2 profiles
  - Example : ./example_data/source_omics1_example_X.csv, source_omics12_example_X.csv
* Integer-converted subtype label for the source dataset (**Source_Y**)
  - The column name should be "subtype", and the rows should be sorted in the same way as the ones in "Source_omics1_X".
  - The subtype label should start from 0
  - Example : ./example_data/source_example_y.csv

### 2. Target multi-omics dataset
* Unlabeled multi-omics profiles to be used as target domain (**Target_omics1_X, Target_omics2_X**)
   - Row : Sample, Column : Feature 
   - The first column should have "sample_id" and the last two coulmns shoud be "batch" and "domain_idx" which contain the batch name (string) and integer number (index) discriminating each dataset. Samples in the same dataset should have same number and "domain_idx" should start from 1. You can have multiple batch inside.
   - The first row should contain the feature names.
   - Samples should be sorted in the same order for both omics 1 and omics 2 profiles and batch number should be also same for both omics profiles. For example, if you set "batch_A" as 1 for "domain_idx" in omics 1 profiles, then "batch_A" should have same number 1 as "domian_idx" in omics 2 profiles.
   - Example : ./example_data/target_omics1_example_X.csv, target_omics2_example_X.csv
 
### 3. Meta data
* Category information
  - This information will be used to convert each integer subtype label with the acutual category name (e.g., cell type, cancer subtype) and provide the prediction result for targe dataset.
  - This file should have two columns named with "subtype" and "subtype_int", which indicate the actual subtype name and the integer-converted subtype label you provided in "Source_Y".
  - Example : ./example_data/subtype_category_info.csv
 
* Batch information
  - This information will be used to convert each batch integer index with the acutual batch name and provide the prediction result for targe dataset.
  - This file should have two columns named with "batch" and "domain_idx", which indicate the actual batch name and the integer-converted batch index you provided in "Target_omics1_X".
  - Example : ./example_data/batch_category_info.csv
 
## How to run (Example)
1. Clone the respository, move to the cloned directory, and edit the **run_MoDAmix.sh** to make sure each variable indicate the corresponding files.
2. Run the below command :
```
chmod +x run_MoDAmix.sh
./run_MoDAmix.sh
```
If you clone the directory and run the above command directly, you will get the result for the example dataset.

3. All the results will be saved in the newly created **results** directory.
   * batch_corrected features.csv : multi-omics integrated batch corrected features for both source and target dataset
   * results_target_prediction.csv : predicted subtype label for each sample in target dataset

## Contact
If you have any questions or problems, please contact to joungmin AT vt.edu
