#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate

id=23
python3 main.py --config=DQN_AAS --env-config=animarl_silkmoth --test=False --cont=data with seed=$id --cond=48

