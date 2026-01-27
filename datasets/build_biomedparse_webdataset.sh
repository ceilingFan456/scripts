#!/bin/bash

## Build WebDataset for BiomedParse dataset
## --max_total_samples is for the number of samples to build the whole thing. -1 means use all samples.
python /home/t-qimhuang/scripts/datasets/build_biomedparse_webdataset_improved.py \
    --dataset_dir /home/t-qimhuang/disk/datasets/BiomedParseData \
    --split train \
    --caption_dir train_caption \
    --out_dir wds_200_train \
    --max_total_samples 200 


python /home/t-qimhuang/scripts/datasets/build_biomedparse_webdataset_improved.py \
    --dataset_dir /home/t-qimhuang/disk/datasets/BiomedParseData \
    --split test \
    --caption_dir test_caption \
    --out_dir wds_200_test \
    --max_total_samples 200 