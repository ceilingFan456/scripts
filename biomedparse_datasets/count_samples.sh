#!/bin/bash

BASE_DIR="/home/t-qimhuang/disk/datasets/BiomedParseData"

find "$BASE_DIR" -type d -name "train_mask" | while read train_dir; do
    
    parent_folder=$(basename "$(dirname "$train_dir")")
    test_dir="$(dirname "$train_dir")/test_mask"
    
    if [ -d "$test_dir" ]; then
        train_count=$(find "$train_dir" -type f | wc -l)
        test_count=$(find "$test_dir" -type f | wc -l)
        
        echo "$parent_folder, training: $train_count, testing: $test_count"
    fi
done