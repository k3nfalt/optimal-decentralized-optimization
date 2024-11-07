#!/bin/bash
# MODEL_TYPE=square_one_minus_square
POINT_X=0.1
POINT_Y=1

for MODEL_TYPE in square_one_minus_square symmetric_logistic_loss sqrt_one_plus; do
    GD_MODEL=iac_auto_toy_gd_${MODEL_TYPE}_${POINT_X}_${POINT_Y}
    python3.7 ./code/distributed_optimization_library/experiments/local_optimization_pytorch/config_toy_nn.py --dumps_path /home/tyurina/local_experiments --experiments_name $GD_MODEL --step_sizes 10.0 5.0 1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 --optimizer gd --model_type $MODEL_TYPE --save_every 1 --number_of_iterations 1000 --symmetric_logistic_loss_starting_point $POINT_X $POINT_Y
    sh /home/tyurina/local_experiments/$GD_MODEL/source_folder/execute.sh
    ADAGD_MODEL=iac_auto_toy_adaptive_stable_gd_${MODEL_TYPE}_${POINT_X}_${POINT_Y}
    python3.7 ./code/distributed_optimization_library/experiments/local_optimization_pytorch/config_toy_nn.py --dumps_path /home/tyurina/local_experiments --experiments_name $ADAGD_MODEL --step_sizes 100 --optimizer adaptive_stable_gd --model_type $MODEL_TYPE --save_every 1 --number_of_iterations 1000 --symmetric_logistic_loss_starting_point $POINT_X $POINT_Y
    sh /home/tyurina/local_experiments/$ADAGD_MODEL/source_folder/execute.sh;
done