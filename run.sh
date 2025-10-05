#!/bin/bash

# agent
python3 preprocess_animals.py --env agent --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --trainSize 400 # --check_hist
python3 preprocess_animals.py --env agent --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --trainSize 400 --option CF
# see scripts folder

# silkmoth
python3 preprocess_animals.py --env silkmoth --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --trainSize 48 # --check_hist 
python3 preprocess_animals.py --env silkmoth --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --trainSize 48 --option CF 
# see scripts folder

# fly
python3 preprocess_animals.py --env fly --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --trainSize 94 # --check_hist
# see scripts folder

# newt
python3 preprocess_animals.py --env newt --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --trainSize 240 # --check_hist
# see scripts folder

# dragonfly
python3 preprocess_animals.py --env dragonfly --data_path ../AniMARL_data/ --result_path ../AniMARL_results/preprocessed/ --check_hist
# see scripts folder
