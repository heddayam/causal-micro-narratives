#!/bin/bash

NOW_PATH=/data/now/
# NOW_PATH=/data/now/
EXT_PATH=/data/now/extracted/sources

YEAR=$1
MONTH=$2
DATA=$3
echo "Extracting 20$YEAR"
# EXT_PATH=$EXT_PATH/20$YEAR
# mkdir -p $EXT_PATH
tar -xvf $NOW_PATH/now-$YEAR-$MONTH.tar $DATA-$YEAR-$MONTH.zip
# tar -xvf $NOW_PATH/now-$DATA.tar.gz $DATA-$YEAR-01.zip $DATA-$YEAR-02.zip $DATA-$YEAR-03.zip $DATA-$YEAR-04.zip $DATA-$YEAR-05.zip $DATA-$YEAR-06.zip
# tar -xvf $NOW_PATH/now-$DATA.tar.gz $DATA-$YEAR-07.zip $DATA-$YEAR-08.zip $DATA-$YEAR-09.zip $DATA-$YEAR-10.zip $DATA-$YEAR-11.zip $DATA-$YEAR-12.zip