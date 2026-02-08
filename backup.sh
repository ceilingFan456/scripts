#!/bin/bash

## steps for uploading the backup

## install AzCopy
wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy.tar.gz

## backup home directory 
sudo ./azcopy_linux_amd64_10.31.1/azcopy copy "/home/t-qimhuang/*" "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_31_26/home_backup?<your-SAS-token>" --recursive --put-md5
## back up disk1
sudo ./azcopy_linux_amd64_10.31.1/azcopy copy "/datadisk/*" "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_31_26/disk1_backup?<your-SAS-token>" --recursive --put-md5
## back up disk2
sudo ./azcopy_linux_amd64_10.31.1/azcopy copy "/datadisk2/*" "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_31_26/disk2_backup?<your-SAS-token>" --recursive --put-md5


## steps for download the backup

## install AzCopy
wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy.tar.gz
# Find the folder name (it might be a newer version than your script)
AZDIR=$(ls -d azcopy_linux_amd64_*)

## redownload home directory
./$AZDIR/azcopy copy "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_31_26/home_backup/*?<your-SAS-token>" "/home/t-qimhuang/" --recursive
## redownload disk1
./$AZDIR/azcopy copy "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_31_26/disk1_backup/*?<your-SAS-token>" "/datadisk/" --recursive
## redownload disk2
./$AZDIR/azcopy copy "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_31_26/disk2_backup/*?<your-SAS-token>" "/datadisk2/" --recursive
