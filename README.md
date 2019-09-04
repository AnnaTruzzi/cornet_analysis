# CORnet analysis

Examine usefulness of representations for classification, through development and across layers of network.
Will be used to expand Anna Birbeck's project relating AoA and layer in network

* Examination of development of CORnet-S representations
* Loads a checkpoint that is the result of training
* Freezes weights in convolutional network
* Adds a logistic regression to each layer

## Requirements

imagenet data, with train and val_in_folders directories
GPU machine
launch.json contains arguments for running of eval_linear.py with VSCODE
checkpoints (e.g., from s3://neurana-imaging/rhodricusack/cornet/)

## Description of procedure
(1) Ran CORnet and get models from multiple epochs of training
(2) Ran eval_linear.sh (or eval_linear_2019-08-31.sh) to 
* for each epoch and conv, load model and add logistic regression layer
* train up
* validate
* validate on each aoa category separately (given --aoaval)
(3) Ran summarize_performance.py to summarize, make graphs and do stats

## Outputs
### from CORnet are in $HOME/cornet directory 
### from eval_linear
linearclass: was fine, but didn't record reglog layers
linearclass_v3: main run. Initially, epochs 0,5,10,20,30 analysed. But will add 1,2,3,4,15,25,35 using eval_linear_2019-08-31.sh

Based on code from  DeepCluster from Facebook Research https://github.com/facebookresearch/deepcluster
and CORnet from Dicarlo lab at MIT https://github.com/dicarlolab/CORnet



