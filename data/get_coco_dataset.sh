#!/bin/bash

mkdir images
cd images

# Download Images
wget -c "https://pjreddie.com/media/files/train2014.zip" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/val2014.zip" --header "Referer: pjreddie.com"

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip

cd ..

# Download COCO Metadata
wget -c "https://pjreddie.com/media/files/instances_train-val2014.zip" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/coco/5k.part" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/coco/trainvalno5k.part" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/coco/labels.tgz" --header "Referer: pjreddie.com"
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

rm train2014.zip val2014.zip instances_train-val2014.zip 5k.part trainvalno5k.part