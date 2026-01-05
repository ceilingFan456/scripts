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

## default and test currently works.
egoexo -o /storage/qiming/datasets/ego-exo4d_data --s3_profile test --parts metadata annotations takes captures take_trajectory

## do not use the --s3_profile flag, not sure why it fails, just set the default profile 
egoexo -o /storage/qiming/datasets/ego-exo4d_data --parts all

## here should be the correct profile to use. it is the one from the email. 
# (ego4d) qiming@showlab02:/storage/qiming/datasets$ aws configure list --profile test
# NAME       : VALUE                    : TYPE             : LOCATION
# profile    : test                     : manual           : --profile
# access_key : ****************2F3Y     : shared-credentials-file : 
# secret_key : ****************lT5y     : shared-credentials-file : 
# region     : ap-southeast-1           : config-file      : ~/.aws/config


## update aws region 
aws configure set region eu-central-1 --profile exo4d


## things to download 
# metadata	0.046	See metadata
# annotations	10.533	All the annotations in Ego-Exo4D
# takes	10553.486	Frame aligned video files associated to the takes
# captures	43.618	Timesync and post-survey data at the capture level (multiple takes)




# take_trajectory	509.503	Trajectories trimmed at each take
# take_eye_gaze	3.265	Eye gaze for each take (3D & 2D)
# take_point_cloud	6164.615	Point clouds for each take
# take_vrs	12301.458	VRS files for each take
# take_vrs_noimagestream	995.592	VRS files for each take without image stream data (video data within MP4 containers with --parts takes)
# capture_trajectory	851.691	Trajectory at the capture-level
# capture_eye_gaze	5.619	Eye gaze at the capture-level (3D)
# capture_point_cloud	4750.039	Point clouds for each capture
# downscaled_takes/448	438.556	Downscaled takes at 448px on the shortest side
# features/omnivore_video	49.986	Omnivore video features
# features/maws_clip_2b	533.826	MAWS CLIP (ViT-2b) features for each frame of video
# ego_pose_pseudo_gt	138.629	Pseudo-ground truth data for Ego Pose
# expert_commentary	42.292	Commentaries for each expert (audio recordings)
# take_transcription	0.094	Audio transcriptions for each take
# take_audio	1056.907	Audio files for the egocentric aria camera
# all	38449.753	All data within the release (you can use --parts all)



# default	12112.778	The default set of data in the release (you can use --parts default or provide no parts)