#!/bin/sh

ROOT_DIR="$(dirname "$0")"
mkdir "$ROOT_DIR/datasets"
cd "$ROOT_DIR/datasets"

# Stanford 40 - dataset
wget http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip
wget http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip

unzip Stanford40_JPEGImages.zip -d Stanford40/
unzip Stanford40_ImageSplits.zip -d Stanford40/

# TV Human Interaction (TV-HI)
wget http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_videos.tar.gz
wget http://www.robots.ox.ac.uk/~alonso/data/readme.txt

mkdir TV-HI
tar -xvf  'tv_human_interactions_videos.tar.gz' -C TV-HI
mv readme.txt 'TV-HI/readme.txt'

rm *.zip
rm *.gz