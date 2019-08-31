# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# Rhodri Cusack Trinity College Dublin, www.cusacklab.org, 2019-08-30

DATA="${HOME}/imagenet"
PYTHON="python"

conda activate pytorch_p36

for TIMEPOINT in 00 01 02 03 04 05  
do
  for CONV in 0 1 2 3
  do

  MODEL="${HOME}/cornet_2019-08-31/epoch_${TIMEPOINT}.pth.tar"
      EXP="${HOME}/cornet_analysis/linearclass_time_${TIMEPOINT}_conv_${CONV}_v3"
      echo "CORnet timepoint ${TIMEPOINT} block ${CONV}"
      echo "${EXP}"
      mkdir -p ${EXP}

      ${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --epochs 1 --conv ${CONV} --lr 0.01 --wd -7 --verbose --exp ${EXP} --workers 32 --aoaval
      # Cannot use tencrops option as the lambda function breaks pickle in multiprocessing.reduction.dump
  done
done