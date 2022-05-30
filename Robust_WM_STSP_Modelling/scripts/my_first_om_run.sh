#!/bin/bash
#SBATCH --job-name=om_test
#SBATCH --gres=gpu:0
#SBATCH --time=0-1:00:00
#SBATCH --mem=750GB
#SBATCH --output=/om2/user/leokoz8/code/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/results/%j.out
#SBATCH -n 20                     # 20 CPU core
#SBATCH -p normal

source activate rwmstsp

MAIN_PATH="/om2/user/leokoz8/code/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/robust_wm_stsp/lightning_main.py"

#python ${MAIN_PATH%.cpp} --epochs 100 --nl 'tanh' --rnn_type 'vRNN' --lr 1e-3

#python ${MAIN_PATH%.cpp} --epochs 100 --nl 'relu' --rnn_type 'vRNN' --lr 1e-3

python ${MAIN_PATH%.cpp} --epochs 100 --nl 'none' --rnn_type 'ah' --lr .01 --hs 500

#python ${MAIN_PATH%.cpp} --epochs 100 --nl 'relu' --rnn_type 'stsp' --lr .02 --hs 1000

#TODO:
    #-Try running all this shit on CPU. Maybe it will be faster. 