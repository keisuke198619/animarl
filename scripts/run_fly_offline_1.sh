#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate

# python3 learn_behavior.py --model RNN --env animarl_fly --data_path ../AniMARL_data --result_path ../AniMARL_results --num_workers 4 --epochs 30 --lr 1e-3 --trainSize 94
# python3 learn_behavior.py --model MLP --env animarl_fly --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --lr 1e-3 --trainSize 94
python3 learn_pretrainQ.py --model DQN_RNN --env animarl_fly --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --num_workers 4 --DIL --lmd2 0.5 --lr 1e-4 --trainSize 94
