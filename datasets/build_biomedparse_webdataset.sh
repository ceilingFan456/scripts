#!/bin/bash

## Build WebDataset for BiomedParse dataset
## --max_total_samples is for the number of samples to build the whole thing. -1 means use all samples.
python build_biomedparse_webdataset_improved.py \
  --root /home/t-qimhuang/disk/datasets/BiomedParseData \
  --split both \
  --caption_dir "{split}_caption" \
  --out_dir "wds_full_{split}_full_caption" \
  --max_total_samples -1 \
  --max_samples_to_save 10 \
  --skip_missing