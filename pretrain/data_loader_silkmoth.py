import math, os
import numpy as np
import torch
from torch.utils import data
import pandas as pd
from utils.utils_chase import get_observation_from_state, discrete_direction

class Dataset(data.Dataset):
    def __init__(self, list_IDs, args, TEST=False):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.data_dir = args.input_data_path
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_all_agents = args.n_all_agents
        self.reward = args.reward
        self.opponent = args.opponent
        self.behavior = args.behavior
        self.Fs = args.Fs
        self.TEST = TEST
        self.batchsize = args.batch_size
        self.len_seqs = len(list_IDs)
        self.args = args
        # self.foldernames = ['Cond1','Cond2','Cond3','Cond4','Cond5','Condi','Condii','Condiii']
        # self.filenames = ['cond1','cond2','cond3','cond4','cond5','condi','condii','condiii']
        
        self.n_files = args.n_files
        self.test_cond = args.test_cond
        self.folders = [0,1,2,3,4]
        '''if self.test_cond is None and not TEST:
            self.folders = [0,1,2,3] # video 1-4
        elif self.test_cond is None and TEST:
            self.folders = [4] # 5
        elif self.test_cond == 'cont_odor' or self.test_cond == 'odor_cont':
            self.folders = [0,1,2,3,4] # 1-5
        self.n_folders = len(self.folders)'''
        self.origin = args.origin
            
    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        data_dir = self.data_dir
        Fs = self.Fs
        max_len = self.args.max_len 

        # Select sample
        if self.list_IDs[index] >= 30:
            cond = 1
            index = self.list_IDs[index] - 30
        else:
            cond = 0
            index = self.list_IDs[index]
        file_index = int(np.floor(index/self.n_files))
        ID = index

        data_dir_video = data_dir + os.sep + "silkmoth_video_"+str(self.folders[file_index])+"_2.0_Hz.npz" # 
        video = np.load(data_dir_video, allow_pickle=True)['arr_0']
        if self.args.option == 'left':
            preprocessed = np.load(os.path.join(self.args.input_data_path,'silkmoth_left_'+str(self.args.trainSize)+'.npz'), allow_pickle=True)
        else:
            preprocessed = np.load(os.path.join(self.args.input_data_path,'silkmoth_'+str(self.args.trainSize)+'.npz'), allow_pickle=True)
        state = preprocessed['arr_0'][ID] # [posx2, velx2, anglex1, odorx2, visionx1, windx4]
        actions = preprocessed['arr_1'][ID]
        reward = preprocessed['arr_2'][ID]
        length = preprocessed['arr_3'][ID]
        conditions = preprocessed['arr_4'][ID]
        us = preprocessed['arr_5']
        ds = preprocessed['arr_6']

        # observation
        observation = state[:,2:][None,:] # remove pos

        # state 
        video_mean = video[:length].mean(1)
        state_ = np.concatenate([state,video_mean.reshape((length,-1))],1)
        # action
        action_onehot = np.identity(13)[actions][:-1]

        if length < max_len:
            state = np.concatenate([state,np.repeat(state[-1:],max_len-length,0)],0)
            observation = np.concatenate([observation,np.repeat(observation[:,-1:],max_len-length,1)],1)
            action_onehot = np.concatenate([action_onehot,np.repeat(action_onehot[-1:],max_len-length,0)],0)

        avail_actions = np.ones_like(action_onehot)
            
        terminated = np.zeros_like(state)[:,:1]
        terminated[length-1] = 1
        filled = np.zeros_like(state)[:,:self.n_all_agents]
        filled[:length] = 1      
        

        if not self.behavior:
            # to tensor
            reward = torch.from_numpy(reward.astype(np.float32)) # scalar
            avail_action = torch.from_numpy(avail_actions.transpose((1,0,2)).astype(np.float32)) # agents,time,dim

        # to tensor
        observation = torch.from_numpy(observation.astype(np.float32)) # agents,time,dim
        action = torch.from_numpy(action_onehot.transpose((1,0,2)).astype(np.float32)) # agents,time,dim
        state = torch.from_numpy(state.astype(np.float32)) # time,dim
        length = torch.from_numpy(np.array(length).astype(np.float32)) # scalar
        terminated = torch.from_numpy(terminated.astype(np.float32)) # time,1
        filled = torch.from_numpy(filled.astype(np.float32)) # time,agents
        if self.behavior:
            return observation,state,action,terminated,filled,length
        else:
            return observation,state,action,avail_action,reward,terminated,filled,length

