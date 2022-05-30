#!/bin/bash
#SBATCH --job-name=robustness_loop
#SBATCH --gres=gpu:0
#SBATCH --time=0-1:00:00
#SBATCH --mem=128GB
#SBATCH --output=/om2/user/leokoz8/code/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/results/%j.out
#SBATCH -n 1000                     # 20 CPU core
#SBATCH --qos=millerlab
#SBATCH --partition=millerlab

source activate rwmstsp

MAIN_PATH="/om2/user/leokoz8/code/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/scripts/do_structural_robustness_analysis.py"

python ${MAIN_PATH%.cpp}