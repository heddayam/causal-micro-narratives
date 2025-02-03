#!/bin/bash

NOW_PATH=/data/now/tar
EXT_PATH=/data/now/extracted

YEAR=14
echo "Extracting 20$YEAR"
EXT_PATH=/data/now/extracted/20$YEAR
mkdir -p $EXT_PATH
tar -xvf $NOW_PATH/now-text.tar.gz text-$YEAR-01.zip text-$YEAR-02.zip text-$YEAR-03.zip text-$YEAR-04.zip text-$YEAR-05.zip text-$YEAR-06.zip
tar -xvf $NOW_PATH/now-text.tar.gz text-$YEAR-07.zip text-$YEAR-08.zip text-$YEAR-09.zip text-$YEAR-10.zip text-$YEAR-11.zip text-$YEAR-12.zip