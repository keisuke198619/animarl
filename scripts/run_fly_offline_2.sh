#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate

python3 learn_pretrainQ.py --model DQN_RNN --env animarl_fly --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --num_workers 4 --AS --lmd2 10 --lr 1e-3 --trainSize 94
# python3 learn_pretrainQ.py --model DQN --env animarl_fly --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --num_workers 4 --AS --lmd2 10 --lr 1e-3 --trainSize 94
#️ python3 learn_pretrainQ.py --model DQN --env animarl_fly --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --num_workers 4 --DIL --lmd2 1 --lr 1e-4 --trainSize 94
