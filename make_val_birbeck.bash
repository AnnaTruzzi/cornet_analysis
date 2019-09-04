#!/bin/bash

# Split imagenet validation into two folders - one with birbeck selected words and one not
# Rhodri Cusack 2019-09-03 www.cusacklab.org

inpth=$HOME/imagenet/val_not_birbeck
outpth=$HOME/imagenet/val_birbeck

while IFS=, read -r col1 col2 col3 col4
do
    # Check if frequency present
    if [ -z "$col4" ]; then
        echo "Empty $col1"
    else
        mv $inpth/$col1 $outpth
    fi
    
done < $HOME/cornet_analysis/matchingAoA_ImageNet_excel.csv

