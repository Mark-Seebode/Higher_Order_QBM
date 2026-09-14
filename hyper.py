import argparse
import shutil
from collections import defaultdict
import time
import subprocess
import sklearn.utils
from malt.pyct.common_transformers.anf import LEAVE
from sklearn.metrics import (accuracy_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tests.devices.qubit_mixed.test_qubit_mixed_measure import BATCH_SIZE
from tqdm import tqdm

import wandb
import random
import numpy as np
from functools import partial
import pickle
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import os

from QBM import QBM


def get_averages(list_of_lists):
    array_of_arrays = np.array(list_of_lists)
    averages = np.mean(array_of_arrays, axis=0)
    return averages




def configure_hyperparams(run):
    global BATCH_SIZE
    global LEARNING_RATE
    #global CHEBYSHEV_DEGREE
    global BETA
    #global GV_1
    #global GV_2

    if run:

        config_defaults = {'batch_size': args.batch_size,  'learning_rate': args.learning_rate,
                             'beta': args.beta,}
                           #'gv_1': 0.1}# 'gv_2': 0.1}

        run.config.setdefaults(config_defaults)

        BATCH_SIZE = wandb.config.batch_size
        LEARNING_RATE = wandb.config.learning_rate
        #CHEBYSHEV_DEGREE = wandb.config.chebyshev_degree
        BETA = wandb.config.beta
        #GV_1 = wandb.config.gv_1
        #GV_2 = wandb.config.gv_2
    else:
        # Set default hyperparameters
        BATCH_SIZE = args.batch_size
        LEARNING_RATE = args.learning_rate
        #CHEBYSHEV_DEGREE = args.chebyshev_degree
        BETA = args.beta
        #GV_1 = 1.0
        #GV_2 = 1.0

    return ("_b" + str(BATCH_SIZE) + "_l" + str(LEARNING_RATE) + "_cd_no"  + "_bt" + str(BETA))
            #+ "_gv1_" + str(GV_1)) #+ "_gv2_" + str(GV_2))


def load_acc_auc_list( seed, new_run_path):
    with open(new_run_path  + f"/acc_list_seed{seed}.pkl", "rb") as f:
        acc_list = pickle.load(f)
    with open(new_run_path + f"/auc_list_seed{seed}.pkl", "rb") as f:
        auc_list = pickle.load(f)
    return acc_list, auc_list


def run_slurm_with_hyperparams(seed, path):
    """Submits a SLURM job with hyperparameters and waits for it to complete."""

    shell_script = "slurm_hyperparam_single_seed.sh"
    slurm_command = [
        "sbatch", shell_script,
        str(seed),
        str(BATCH_SIZE),
        str(LEARNING_RATE),
        #str(CHEBYSHEV_DEGREE),
        str(BETA),
        #str(GV_1),
        #str(GV_2),
        str(path)
    ]

    print("Running running slurm command")
    job_submission = subprocess.run(slurm_command, capture_output=True, text=True)
    print(f"Submitted SLURM job:\n{job_submission.stdout}")
    print(f"Error (if any):\n{job_submission.stderr}")

    # Extract SLURM job ID
    job_id = job_submission.stdout.strip().split()[-1]  # Last word is the job ID
    return job_id

def wait_for_slurm_jobs(job_ids):
    """Wait for all SLURM jobs to finish before proceeding."""
    print(f"Waiting for SLURM jobs to finish: {job_ids}")
    while True:
        cmd = f"squeue -u seebode -h -o \"%i\""
        out = subprocess.getoutput(cmd)
        running = out.split()
        if not any(jid in running for jid in job_ids):
            print("All SLURM jobs have completed.")
            break
        time.sleep(10)




def main(args, resume=False, resume_id="6wv4w77p"):
    print("Starting current sweep")

    # start run
    if HYPERPARAM_OPT:
        if resume:
            run = wandb.init(
                project="HOQBM_UCI",
                entity="seebode-mark-ludwig-maximilianuniversity-of-munich",
                group=SWEEP_ID, id=resume_id, resume="must")
        else:
            run = wandb.init(reinit=True, group=SWEEP_ID)

    else:
        run = None



    params_string_for_run = configure_hyperparams(run)
    print("Params string for run: ", params_string_for_run)

    if HYPERPARAM_OPT:
        run.name = params_string_for_run

        print("Run name: ", run.name)

        num_metrics=2
        metrics_for_all_seeds = [[] for i in range(num_metrics)]
        job_list = []


        seeds = args.seeds

        epoch_data = defaultdict(lambda: {
            'acc_val': [],
        })

        new_run_path = args.path
        os.makedirs(new_run_path, exist_ok=True)

        for seed in seeds:
            job_id = run_slurm_with_hyperparams(seed, new_run_path)
            job_list.append(job_id)
        print(job_list)
        time.sleep(10)
        wait_for_slurm_jobs(job_list)
        time.sleep(10)

        for seed in seeds:
            acc_list_file = args.path + f"acc_list_seed{seed}.pkl"
            if os.path.exists(acc_list_file):
                with open(acc_list_file, "rb") as f:
                    acc_list = pickle.load(f)

                for epoch in range(20):
                    epoch_data[epoch]['acc_val'].append(acc_list[epoch])

            else:
                raise FileNotFoundError(f"Result file not found: {acc_list_file}")

        folder_path = os.path.dirname(args.path)
        shutil.rmtree(folder_path)

        epochs_sorted = sorted(epoch_data.keys())
        avg_acc_list = [np.mean(epoch_data[e]['acc_val']) for e in epochs_sorted]


        best_epoch = int(np.argmax(avg_acc_list))

        best_acc = avg_acc_list[best_epoch]

        metrics_for_all_seeds[0].append(best_acc)
        metrics_for_all_seeds[1].append(best_epoch)
        print(f"Loaded results: ACC={best_acc} at epoch={best_epoch}")
        print("All seeds finished")

        if HYPERPARAM_OPT:
            for metric_index in range(len(metrics_for_all_seeds)):
                metrics_for_all_seeds[metric_index] = get_averages(metrics_for_all_seeds[metric_index])

            wandb.log({"acc": metrics_for_all_seeds[0], "best_epoch": metrics_for_all_seeds[1]})

            run.finish()

        print("Run finished.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get optimmal cross entropy weights for chestmnist')


    parser.add_argument('-lr', '--learning_rate',
                        metavar='FLOAT',
                        help='Learning rate for the optimizer',
                        default=0.001,
                        type=float)

    parser.add_argument('-e', '--epochs',
                        metavar='INT',
                        help='Number of epochs to train for',
                        default=20,
                        type=int)

    parser.add_argument('-b', '--batch_size',
                        metavar='INT',
                        help='Batch size for training',
                        default=25,
                        type=int)


    parser.add_argument('-s', '--seed',
                        metavar='INT',
                        help='Seed for RNG',
                        default=42,
                        type=int)

    parser.add_argument('-nr', '--n_runs',
                        metavar='INT',
                        help='Number of runs to perform',
                        default=1,
                        type=int)

    parser.add_argument('-hpo', '--hyperparam_opt',
                        metavar='BOOL',
                        help='Whether to perform hyperparameter optimization',
                        default=True,
                        type=bool)

    parser.add_argument('--n_sweeps',
                        metavar='INT',
                        help='Number of sweeps to perform',
                        default=50,
                        type=int)

    parser.add_argument('--path',
                        metavar='STR',
                        help='Path to save the results',
                        default="out/",
                        type=str)

    parser.add_argument('--chebyshev_degree',
                        metavar='INT',
                        help='Degree of Chebyshev polynomial for approximating the exponential in the QBM training',
                        default=2,
                        type=int)

    parser.add_argument('--beta',
                        metavar='FLOAT',
                        help='Beta parameter for the QBM training',
                        default=3.0,
                        type=int)

    parser.add_argument('--sweep_id', type=str, default="")
    parser.add_argument('--sweep_path', type=str, default="")
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[])

    args = parser.parse_args()

    print("Starting Hyperparameter Optimization for")

    HYPERPARAM_OPT = args.hyperparam_opt

    if HYPERPARAM_OPT:
        if args.key:
            wandb.login(key=args.key)
        else:
            with open("wandb_key.txt", "r") as f:
                key = f.read().strip()
            wandb.login(key=key)
        print("Logged in to wandb")

        SWEEP_ID = args.sweep_id
        SWEEP_PATH = args.sweep_path

        sweep_id_path = SWEEP_PATH + SWEEP_ID
        print(sweep_id_path)
        main_with_args = partial(main, args)
        print("Starting sweeping")
        wandb.agent(sweep_id=sweep_id_path, function=main_with_args,
                    count=50)

    else:
        main(args)







