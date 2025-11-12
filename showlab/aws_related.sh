#!/bin/bash

## check the contents of the ego4d-speac s3 bucket
aws s3 ls s3://ego4d-speac/

## download the content of speac dataset
aws s3 sync s3://ego4d-speac/ ./ego4d_data/ --progress-frequency 60

## check the size of the data to be downloaded
aws s3 ls s3://ego4d-speac/ --recursive --human-readable --summarize

## found some data under this path 
ls /storage/chenjoya/datasets/ego4d

## current content under the ego4d-speac bucket
# (base) qiming@showlab02:/storage/chenjoya/datasets/ego4d$ aws s3 ls s3://ego4d-speac/
#                            PRE canonical/
#                            PRE data/
#                            PRE ego/
#                            PRE egoexo-public/
#                            PRE egoexo/
#                            PRE exo/
#                            PRE metadata-v2/
#                            PRE metadata_v0/
#                            PRE metadata_v1.1/
#                            PRE metadata_v1.2/
#                            PRE metadata_v1/
#                            PRE mobile/
#                            PRE objects/
#                            PRE public/
#                            PRE social_test_data/
#                            PRE test/
#                            PRE v1/
#                            PRE videos/


## repo to the installer repo
git clone https://github.com/facebookresearch/Ego4d.git

## aws list profiles
aws configure list-profiles

## to see details
aws configure list --profile <profile_name>

## profile name 
# (ego4d) qiming@showlab02:/storage/qiming/datasets$ aws configure list --profile exo4d
# NAME       : VALUE                    : TYPE             : LOCATION
# profile    : exo4d                    : manual           : --profile
# access_key : ****************JBM2     : shared-credentials-file : 
# secret_key : ****************Izkm     : shared-credentials-file : 
# region     : ap-southeast-1           : config-file      : ~/.aws/config

## You can have multiple profiles setup on your machine. To tell the downloader which profile to use, simply use the flag --s3_profile <name>.


# By default, this will download the recommended set of data. This is equivalent to providing --parts metadata annotations takes captures take_trajectory. This is quite large (~14TiB), and as such the rest of this document will describe how to filter down this set or include parts that are not in the "recommended" set.
# egoexo -o <out-dir> --s3_profile exo4d
egoexo -o /storage/qiming/datasets/ego-exo4d_data --s3_profile exo4d --parts all


## update aws region 
aws configure set region eu-central-1 --profile exo4d