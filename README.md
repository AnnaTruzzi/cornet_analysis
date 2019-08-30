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

## Example usage

Based on code from  DeepCluster from Facebook Research https://github.com/facebookresearch/deepcluster
and CORnet from Dicarlo lab at MIT https://github.com/dicarlolab/CORnet

