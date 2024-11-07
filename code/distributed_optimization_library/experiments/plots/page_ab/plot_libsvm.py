import os

def main():
    experiments = [("page_ab_neurips_2022_australian_logistic_theoretical_step", 10**6),
                   ("page_ab_neurips_2022_covtype_logistic_theoretical_step", 10**6), 
                   ("page_ab_neurips_2022_mushrooms_logistic_theoretical_step", 10**6), 
                   ("page_ab_neurips_2022_real-sim_logistic_theoretical_step", 0.1),
                   ("page_ab_neurips_2022_w8a_logistic_theoretical_step", 1.0)]
    for exp, max_show in experiments:
        os.system("python3 code/distributed_optimization_library/experiments/plots/page_ab/plot_vr-marina_mini-batch.py --dumps_paths ~/exepriments/{exp} --output_path ~/tmp_launch/{exp} --max_show {max_show}".format(exp=exp, max_show=max_show))

if __name__ == "__main__":
    main()