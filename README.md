# "On the Optimal Time Complexities in Decentralized Stochastic Asynchronous Optimization" *Alexander Tyurin, Peter Richtárik*
## The code to reproduce the experiments from the *accepted paper at NeurIPS 2024*

## Quick Start
### 1. Install [Singularity](https://sylabs.io/guides/3.5/user-guide/introduction.html) (optional)
If you don't want to install Singularity, make sure that you have all dependecies from Singularity.def (python3, numpy, pytorch, etc.)

a. Pull an image 
````
singularity pull library://k3nfalt/default/python_ml:sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745
````
b. Open a shell console of the image
````
singularity shell --nv ~/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif
````
### 2. Prepare scripts for experiments

````
mkdir ~/exepriments/
````

````
source prepare.sh && python3.7 code/distributed_optimization_library/experiments/asynchronous/config_async.py --dataset_path ~/data --dumps_path ~/exepriments --experiments_name EXPERIMENTS_NAME --num_nodes 100 --step_size_range -10 10 --number_of_iterations 100000000 --algorithm_names minibatch_sgd fragile --delays equal --quality_check_rate 1 --not_only_intel --fixed_seed 4343 --task quadratic --quad_dim 1000 --quad_noise_lambda 0.0 --quad_strongly_convex_constant 1e-6 --batch_size 1 --gb_per_task 60 --stop_time 1000000 --mkl_fix --scale_quality_check_rate 10 --rennala_page_inner_batch_size_full 1 --rennala_batch_size_list 40 80 120 160 200 300 400 500 --graph torus --graph_dist_scale 0.1 --quad_noise 0.001 --quad_type_noise zero-last
````

````
source prepare.sh && python3.7 code/distributed_optimization_library/experiments/asynchronous/config_async.py --dataset_path ~/data --dumps_path ~/exepriments --experiments_name EXPERIMENTS_NAME --num_nodes 100 --step_size_range -10 10 --number_of_iterations 100000000 --algorithm_names minibatch_sgd fragile --delays equal --quality_check_rate 1 --not_only_intel --fixed_seed 4343 --task quadratic --quad_dim 1000 --quad_noise_lambda 0.0 --quad_strongly_convex_constant 1e-6 --batch_size 1 --gb_per_task 60 --stop_time 1000000 --mkl_fix --scale_quality_check_rate 10 --rennala_page_inner_batch_size_full 1 --rennala_batch_size_list 40 80 120 160 200 300 400 500 --graph torus --graph_dist_scale 1.0 --quad_noise 0.001 --quad_type_noise zero-last
````

````
source prepare.sh && python3.7 code/distributed_optimization_library/experiments/asynchronous/config_async.py --dataset_path ~/data --dumps_path ~/exepriments --experiments_name EXPERIMENTS_NAME --num_nodes 100 --step_size_range -10 10 --number_of_iterations 100000000 --algorithm_names minibatch_sgd fragile --delays equal --quality_check_rate 1 --not_only_intel --fixed_seed 4343 --task quadratic --quad_dim 1000 --quad_noise_lambda 0.0 --quad_strongly_convex_constant 1e-6 --batch_size 1 --gb_per_task 60 --stop_time 1000000 --mkl_fix --scale_quality_check_rate 10 --rennala_page_inner_batch_size_full 1 --rennala_batch_size_list 40 80 120 160 200 300 400 500 --graph torus --graph_dist_scale 10.0 --quad_noise 0.001 --quad_type_noise zero-last
````


### 3. Execute scripts
````
sh ~/exepriments/EXPERIMENTS_NAME/singularity_*.sh
````

