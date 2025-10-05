#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate

id=25
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_agent_2vs1 --test=False --cont=data with seed=$id --cond=400
