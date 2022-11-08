#!/bin/bash


# Place all blobs in the ./blobs directory
mkdir -p blobs

wget https://pjreddie.com/media/files/yolov3.weights -O ./blobs/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O ./blobs/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O ./blobs/coco.names
wget https://github.com/OlafenwaMoses/ImageAI/raw/master/data-videos/traffic.mp4 -O ./blobs/traffic.mp4