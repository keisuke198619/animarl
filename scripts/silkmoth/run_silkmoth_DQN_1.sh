#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/venv/envs/animarl_A6000/bin/activate

id=21
python3 main.py --config=DQN --env-config=animarl_silkmoth --test=False --cont=False with seed=$id --cond=48