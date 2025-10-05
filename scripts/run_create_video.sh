#!/bin/bash

cd /home/fujii/workspace3/work/AniMARL-dev
source $HOME/workspace3/venvs/gfootball2/bin/activate

python create_video.py --result_file DQN/25_RNN_pretrain_from_demo_DIL-ts-400 --env animarl_agent --output DIL1 --seed 5 --index 24
python create_video.py --result_file DQN/25_RNN_pretrain_from_demo_DIL-ts-CF5 --env animarl_agent --output DIL1_2  --seed 5 --index 24
python create_video.py --result_file DQN/25_RNN_pretrain_from_demo_DIL-ts-CF --env animarl_agent --output CF1 --seed 5 --index 24
python create_video.py --result_file DQN/25_RNN_pretrain_from_demo_DIL-ts-CF6 --env animarl_agent --output CF1_2  --seed 5 --index 24

python create_video.py --result_file DQN/22_RNN_from_demo_DIL-ts-48 --env animarl_silkmoth --output DIL2 --seed 2 --index 50
python create_video.py --result_file DQN/22_RNN_from_demo_DIL-ts-CF3 --env animarl_silkmoth --output DIL2_1  --seed 2 --index 50
python create_video.py --result_file DQN/23_RNN_from_demo_DIL-ts-CF2 --env animarl_silkmoth --output CF2 --seed 3 --index 50
python create_video.py --result_file DQN/23_RNN_from_demo_DIL-ts-CF4 --env animarl_silkmoth --output CF2_1 --seed 3 --index 50 

#python create_video.py --result_file DQN/24_RNN_pretrain_from_demo_DIL-ts-400 --env animarl_agent --output DIL1 --seed 4 --index 8
#python create_video.py --result_file DQN/24_RNN_pretrain_from_demo_DIL-ts-CF5 --env animarl_agent --output DIL1_2  --seed 4 --index 58
#python create_video.py --result_file DQN/21_RNN_pretrain_from_demo_DIL-ts-CF --env animarl_agent --output CF1 --seed 1 --index 8
#python create_video.py --result_file DQN/22_RNN_pretrain_from_demo_DIL-ts-CF6 --env animarl_agent --output CF1_2  --seed 2 --index 8
<< COMMENT

python create_video.py --result_file DQN/24_RNN_from_demo_DIL-ts-400 --env animarl_agent --output DIL2 --seed 4 --index 8
python create_video.py --result_file DQN/24_RNN_from_demo_DIL-ts-CF3 --env animarl_agent --output DIL2_1  --seed 4 --index 58
python create_video.py --result_file DQN/21_RNN_from_demo_DIL-ts-CF2 --env animarl_agent --output CF2 --seed 1 --index 8
python create_video.py --result_file DQN/21_RNN_from_demo_DIL-ts-CF4 --env animarl_agent --output CF2_1 --seed 2 --index 8


python create_video.py --result_file DQN/23_RNN_from_demo_DIL-ts-400 --env animarl_agent --output DIL1 --seed 3 --index 120
python create_video.py --result_file DQN/22_RNN_from_demo_DIL-ts-CF3 --env animarl_agent --output DIL1_2  --seed 2 --index 120
python create_video.py --result_file DQN/24_RNN_from_demo_DIL-ts-CF --env animarl_agent --output CF1 --seed 4 --index 120
python create_video.py --result_file DQN/25_RNN_from_demo_DIL-ts-CF4 --env animarl_agent --output CF1_2 --seed 5 --index 220

python create_video.py --result_file DQN/22_RNN_from_demo_DIL-ts-48 --env animarl_silkmoth --output DIL2 --seed 2 --index 96
python create_video.py --result_file DQN/21_RNN_from_demo_DIL-ts-CF3 --env animarl_silkmoth --output DIL2_1  --seed 4 --index 4
python create_video.py --result_file DQN/21_RNN_from_demo_DIL-ts-CF2 --env animarl_silkmoth --output CF2 --seed 2 --index 50
python create_video.py --result_file DQN/21_RNN_from_demo_DIL-ts-CF4 --env animarl_silkmoth --output CF2_1 --seed 2 --index 50
COMMENT
# << COMMENT
# COMMENT
