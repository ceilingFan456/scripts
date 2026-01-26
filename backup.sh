#!/bin/bash

## install AzCopy
wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy.tar.gz

## backup home directory 
sudo ./azcopy_linux_amd64_10.31.1/azcopy copy "/home/t-qimhuang/*" "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_25_26/home_backup?<your-SAS-token>" --recursive --put-md5

## back up disk1
sudo ./azcopy_linux_amd64_10.31.1/azcopy copy "/datadisk/*" "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_25_26/disk1_backup?<your-SAS-token>" --recursive --put-md5

## back up disk2
sudo ./azcopy_linux_amd64_10.31.1/azcopy copy "/datadisk2/*" "https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_01_25_26/disk2_backup?<your-SAS-token>" --recursive --put-md5