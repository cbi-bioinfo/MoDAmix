#!/bin/bash

data_dir="./example_data/"

source_omics1_filename="source_omics1_example_X.csv"
source_omics2_filename="source_omics2_example_X.csv"
source_class_info_filename="source_example_y.csv"

target_omics1_filename="target_omics1_example_X.csv"
target_omics2_filename="target_omics2_example_X.csv"

python3 moDAmix.py $data_dir $source_omics1_filename $source_omics2_filename $source_class_info_filename $target_omics1_filename $target_omics2_filename