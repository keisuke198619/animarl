#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate
<< COMMENT
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=21 --cond=CF3
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=22 --cond=CF3
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=23 --cond=CF3
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=24 --cond=CF3
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=25 --cond=CF3

python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=21 --cond=CF4
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=22 --cond=CF4
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=23 --cond=CF4
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=24 --cond=CF4
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=25 --cond=CF4

python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=21 --cond=CF5
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=22 --cond=CF5
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=23 --cond=CF5
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=24 --cond=CF5
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=25 --cond=CF5

python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=21 --cond=CF6
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=22 --cond=CF6
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=23 --cond=CF6
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=24 --cond=CF6
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=25 --cond=CF6
COMMENT


python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=21 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=22 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=23 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=24 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=data with seed=25 --cond=48
<< COMMENT
python3 main.py --config=BC --env-config=animarl_silkmoth --test=True --cont=data with seed=21 --cond=48
python3 main.py --config=BC --env-config=animarl_silkmoth --test=True --cont=data with seed=22 --cond=48
python3 main.py --config=BC --env-config=animarl_silkmoth --test=True --cont=data with seed=23 --cond=48
python3 main.py --config=BC --env-config=animarl_silkmoth --test=True --cont=data with seed=24 --cond=48
python3 main.py --config=BC --env-config=animarl_silkmoth --test=True --cont=data with seed=25 --cond=48

python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=21 --cond=CF
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=22 --cond=CF
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=23 --cond=CF
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=24 --cond=CF
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=25 --cond=CF

python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=21 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=22 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=23 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=24 --cond=48
python3 main.py --config=DQN_RNN_DIL --env-config=animarl_silkmoth --test=True --cont=False with seed=25 --cond=48

python3 main.py --config=DQN_RNN --env-config=animarl_silkmoth --test=True --cont=False with seed=21 --cond=48
python3 main.py --config=DQN_RNN --env-config=animarl_silkmoth --test=True --cont=False with seed=22 --cond=48
python3 main.py --config=DQN_RNN --env-config=animarl_silkmoth --test=True --cont=False with seed=23 --cond=48
python3 main.py --config=DQN_RNN --env-config=animarl_silkmoth --test=True --cont=False with seed=24 --cond=48
python3 main.py --config=DQN_RNN --env-config=animarl_silkmoth --test=True --cont=False with seed=25 --cond=48

python3 main.py --config=DQN_RNN_AAS --env-config=animarl_silkmoth --test=True --cont=data with seed=21 --cond=48
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_silkmoth --test=True --cont=data with seed=22 --cond=48
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_silkmoth --test=True --cont=data with seed=23 --cond=48
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_silkmoth --test=True --cont=data with seed=24 --cond=48
python3 main.py --config=DQN_RNN_AAS --env-config=animarl_silkmoth --test=True --cont=data with seed=25 --cond=48
COMMENT