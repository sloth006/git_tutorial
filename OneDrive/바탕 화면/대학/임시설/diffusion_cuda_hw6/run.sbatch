#!/bin/bash

#SBATCH -J diff_inf
#SBATCH -o logs/run_out.diff_inf.%j
#SBATCH -e logs/run_err.diff_inf.%j
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:tegra:1


sh scripts/run.sh
