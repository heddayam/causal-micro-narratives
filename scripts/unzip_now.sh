#!/bin/bash

# NOW_PATH=/data/now/tar
NOW_PATH=/data/now/
EXT_PATH=/data/now/extracted/sources

YEAR=$1
# mkdir -p $EXT_PATH/20$YEAR/zips
# mv $EXT_PATH/20$YEAR/text*zip $EXT_PATH/20$YEAR/zips
echo "Unzipping 20$YEAR"
# EXT_PATH=/data/now/extracted/20$YEAR
# mkdir -p $EXT_PATH/text

months=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")

# Iterate through each month in the list
for month in "${months[@]}"; do
    # Your code to process each month goes here
    # unzip -j $EXT_PATH/zips/text-$YEAR-$month.zip '*us*' -d $EXT_PATH/text
     unzip -j $EXT_PATH/sources-$YEAR-$month.zip
done


# pattern="sources-${YEAR}-*.txt"

# # Define the output file name
# output_file="now-sources-20${YEAR}.txt"

# # Use a loop to concatenate matching files
# for file in $pattern; do
#   cat "$file" >> "$output_file"
# done