#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate


python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=21 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=22 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=23 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=24 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=25 --cond=240

# << COMMENT
python3 main.py --config=BC --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=21 --cond=240
python3 main.py --config=BC --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=22 --cond=240
python3 main.py --config=BC --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=23 --cond=240
python3 main.py --config=BC --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=24 --cond=240
python3 main.py --config=BC --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=25 --cond=240

python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=False with seed=21 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=False with seed=22 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=False with seed=23 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=False with seed=24 --cond=240
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_newt_2vs1 --test=True --cont=False with seed=25 --cond=240

python3 main.py --config=DQN_RNN --env-config=animarl_newt_2vs1 --test=False --cont=data with seed=21 --cond=240
python3 main.py --config=DQN_RNN --env-config=animarl_newt_2vs1 --test=False --cont=data with seed=22 --cond=240
python3 main.py --config=DQN_RNN --env-config=animarl_newt_2vs1 --test=False --cont=data with seed=23 --cond=240
python3 main.py --config=DQN_RNN --env-config=animarl_newt_2vs1 --test=False --cont=data with seed=24 --cond=240
python3 main.py --config=DQN_RNN --env-config=animarl_newt_2vs1 --test=False --cont=data with seed=25 --cond=240

python3 main.py --config=DQN_RNN_AAS --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=21 --cond=240
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=22 --cond=240
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=23 --cond=240
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=24 --cond=240
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_newt_2vs1 --test=True --cont=data with seed=25 --cond=240
# COMMENT
