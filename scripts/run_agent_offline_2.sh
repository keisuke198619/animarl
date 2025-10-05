#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/venv/envs/animarl_A6000/bin/activate

# python3 learn_pretrainQ.py --model DQN_RNN --env animarl_agent --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --num_workers 4 --AS --lmd2 50 --lr 1e-3 --trainSize 400
python3 learn_pretrainQ.py --model DQN --env animarl_agent --data_path ../AniMARL_data --result_path ../AniMARL_results --epochs 30 --num_workers 4 --AS --lmd2 50 --lr 1e-3 --trainSize 400
