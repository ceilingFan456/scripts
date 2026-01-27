#!/bin/bash

# Define the base directory
BASE_DIR="/home/t-qimhuang/disk/datasets/BiomedParseData"

# Check if the directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist."
    exit 1
fi

# Iterate through each subdirectory in the base directory
for dir in "$BASE_DIR"/*/; do
    # Remove trailing slash for the folder name
    parent_folder=$(basename "$dir")
    
    # Define paths to mask directories
    train_mask_dir="${dir}train_mask"
    test_mask_dir="${dir}test_mask"
    
    # Check if BOTH directories exist
    if [ -d "$train_mask_dir" ] && [ -d "$test_mask_dir" ]; then
        # Count files (excluding directories)
        train_count=$(find "$train_mask_dir" -maxdepth 1 -type f | wc -l)
        test_count=$(find "$test_mask_dir" -maxdepth 1 -type f | wc -l)
        
        # Print the results
        echo "$parent_folder, training: $train_count, testing: $test_count"
    fi
done