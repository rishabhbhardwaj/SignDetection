#!/bin/sh

mkdir -p data

wget http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip -P /tmp/
unzip /tmp/TrainIJCNN2013.zip -d data/

mkdir -p data/train
mv data/TrainIJCNN2013/*.ppm data/train/
echo "Data is ready to consume.."

