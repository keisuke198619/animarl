import math, os
import numpy as np
import torch
from torch.utils import data
import pickle
from utils.utils_chase import get_observation_from_state

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
        self.Fs = args.Fs
        self.TEST = TEST
        self.batchsize = args.batch_size
        self.len_seqs = len(list_IDs)
        self.args = args

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def get_dist(self, abs_pos_own, abs_pos_adv):
        pos_rel = abs_pos_adv - abs_pos_own
        dist = np.sqrt(np.sum(np.square(pos_rel),1))
        return dist

    def get_sub_acc(self, abs_pos_own, abs_pos_adv, abs_acc):
        pos_rel = abs_pos_adv - abs_pos_own
        theta = np.arctan2(pos_rel[:,1], pos_rel[:,0])
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) # inverse
        sub_acc = []
        for t in range(abs_acc.shape[0]):
            try: sub_acc.append(np.dot(rot[:,:,t], abs_acc[t,:]))
            except: import pdb; pdb.set_trace()
        return np.array(sub_acc)

    def discrete_direction(self, pos_1, pos_2, vec, no_angles=12):
        vec_projected = self.get_sub_acc(pos_1, pos_2, vec)
        angles = np.arctan2(vec_projected[:,1], vec_projected[:,0])

        if no_angles == 12:
            # relative coordinates 
            # ang = action * -np.pi / 6 # 0:top (toward target), 3:right, 6:bottom, 9:left
            # else: # absolute coordinates 
            width = np.pi/12
            discrete_angles = []
            for angle in angles:
                if angle > -width and angle <= width: # top
                    discrete_angle = 0
                elif angle > -np.pi/6-width and angle <= -np.pi/6+width:  
                    discrete_angle = 1
                elif angle > -np.pi/3-width and angle <= -np.pi/3+width: 
                    discrete_angle = 2
                elif angle > -np.pi/2-width and angle <= -np.pi/2+width: # right
                    discrete_angle = 3
                elif angle > -4*np.pi/6-width and angle <= -4*np.pi/6+width: 
                    discrete_angle = 4
                elif angle > -5*np.pi/6-width and angle <= -5*np.pi/6+width: 
                    discrete_angle = 5
                elif angle <= -np.pi+width or angle > np.pi-width: # bottom
                    discrete_angle = 6
                elif angle >= 5*np.pi/6-width and angle < 5*np.pi/6+width: #  
                    discrete_angle = 7
                elif angle >= 4*np.pi/6-width and angle < 4*np.pi/6+width: # 
                    discrete_angle = 8
                elif angle >= np.pi/2-width and angle < np.pi/2+width: # left
                    discrete_angle = 9
                elif angle >= np.pi/3-width and angle < np.pi/3+width: # 
                    discrete_angle = 10
                elif angle >= np.pi/6-width and angle < np.pi/6+width: # 
                    discrete_angle = 11
                discrete_angles.append(discrete_angle)
        else:
            print('not defined other than no_angles == 12')
            import pdb; pdb.set_trace()
        return np.array(discrete_angles)

    def calculate_acceleration(self, next_vel, vel, damping, speed, Fs):
        # original equation: next_vel = vel * (1 - damping) + acc * speed / Fs
        acc = (next_vel - vel * (1 - damping)) / (speed / Fs)
        # acc_direction = np.arctan2(acc[:,1], acc[:,0])
        return acc

    def __getitem__(self, index):
        '''Generates one sample of data'''
        data_dir = self.data_dir # '../AniMARL_results/preprocessed/'   
        Fs = self.Fs
        max_len = self.args.max_len

        # Select sample
        file_index = int(np.floor(index/100))
        ID = int(np.mod(index,100))
        
        debug = False # only synthetic (agent) data
        if debug:
            rep = np.load('../AniMARL_data/agent/pos_val_rep_2on1_indiv_K_'+str(file_index)+'.npz', allow_pickle=True)
            pos_list = np.array(rep['pos']).squeeze()
            reward_list = np.array(rep['rewards']).squeeze()
            action_list = np.array(rep['actions']).squeeze()
            state_list = np.array(rep['states']).squeeze()  
            actions0 = np.array(action_list[ID])
            us = np.array([3.60001389, 3.60001389, 3.00001667])
            ds = np.array([0.27511111, 0.27669294, 0.27098354])
            state = np.array(state_list[ID])
            length = state.shape[0]
        else:
            if "animarl_agent_2vs1" in self.args.env:
                filename = 'agent'
            elif "fly" in self.args.env:
                filename = 'fly'
            elif "newt" in self.args.env:
                filename = 'newt'

            filename += '_'+str(self.args.trainSize)
            if self.args.option == 'CF':
                filename += '_CF'
            preprocessed = np.load(os.path.join(data_dir,filename+'.npz'), allow_pickle=True)
            state = preprocessed['arr_0'][ID]
            actions = preprocessed['arr_1'][ID]
            reward = preprocessed['arr_2'][ID]
            length = preprocessed['arr_3'][ID]
            conditions = preprocessed['arr_4'][ID]
            us = preprocessed['arr_5']
            ds = preprocessed['arr_6']

        if True:
            if length > 1:
                pos_p1,vel_p1,pos_p2,vel_p2,pos_e,vel_e = state[:,0:2],state[:,2:4],state[:,4:6],state[:,6:8],state[:,8:10],state[:,10:12]
                vel = np.concatenate([vel_p1,vel_p2,vel_e],1)
                # acc = np.diff(vel,axis=0)/self.Fs
                # acc_direction = np.arctan2(acc[:,1:6:2], acc[:,0:6:2])
                acc = []
                for n in range(self.n_all_agents):
                    acc_ = self.calculate_acceleration(vel[1:,n*2:(n+1)*2], vel[:-1,n*2:(n+1)*2], ds[n], us[n], Fs)
                    acc.append(acc_)
                acc = np.array(acc).transpose((1,0,2)).reshape((length-1,-1))

                dist1 = self.get_dist(pos_p1, pos_e)
                dist2 = self.get_dist(pos_p2, pos_e)
                dist = np.concatenate([dist1[:,np.newaxis], dist2[:,np.newaxis]],1)
                adv_index = np.argmin(dist,axis=1)[0]
                abs_pos_adv = np.concatenate([pos_p1[:,:,np.newaxis], pos_p2[:,:,np.newaxis]],2)[:,:,adv_index]
                # p1
                actions_p1 = self.discrete_direction(pos_p1[:-1],pos_e[:-1],acc[:,:2])
                # p2
                actions_p2 = self.discrete_direction(pos_p2[:-1],pos_e[:-1],acc[:,2:4])
                # e1
                actions_e = self.discrete_direction(pos_e[:-1],abs_pos_adv[:-1],acc[:,4:6])

                actions = np.array([actions_p1,actions_p2,actions_e]).transpose()
                actions = np.concatenate([actions,actions[-1:,:]],0) 
                # test: print(actions); print(actions0)

            else: 
                actions = np.array(action_list[ID])
        else:
            actions = np.array(action_list[ID])

        action_onehot = np.identity(13)[actions]
        # action_onehot = np.concatenate([np.zeros((1,action_onehot.shape[1],13)),action_onehot[:-1]],0)
        avail_actions = np.ones_like(action_onehot)
        
        

        # state,action,avail_action,reward,length = states[ID_in_file],actions[ID_in_file],avail_actions[ID_in_file],rewards[ID_in_file],lengths[ID_in_file]
        # missing = missings[ID_in_file]

        if length < max_len:
            action_onehot = np.concatenate([action_onehot,np.repeat(action_onehot[-1:],max_len-length,0)],0)
            avail_actions = np.concatenate([avail_actions,np.repeat(avail_actions[-1:],max_len-length,0)],0)
            state = np.concatenate([state,np.repeat(state[-1:],max_len-length,0)],0)

        terminated = np.zeros_like(state)[:,:1]
        terminated[length-1] = 1
        filled = np.zeros_like(state)[:,:self.n_all_agents]
        filled[:length] = 1      
        #if np.sum(missing[:self.n_agents])>0:
        #    filled[:,np.where(missing[:self.n_agents])[0]] = 0

        # observation
        observation = get_observation_from_state(state,self.n_all_agents,self.n_all_agents-self.n_enemies,self.n_enemies)
        if self.args.env == 'animarl_fly_2vs1':
            observation = np.concatenate([observation,np.sqrt(np.sum(observation[:,:,:2]**2,2))[:,:,None]],2)
        elif self.args.env == 'animarl_agent_2vs1':
            observation = np.concatenate([observation,np.repeat(state[None,:,-1,None], 3, axis=0)],2)

        # to tensor
        observation = torch.from_numpy(observation.astype(np.float32)) # agents,time,dim
        state = torch.from_numpy(state.astype(np.float32)) # time,dim

        action = torch.from_numpy(action_onehot.transpose((1,0,2)).astype(np.float32)) # agents,time,dim
        avail_action = torch.from_numpy(avail_actions.transpose((1,0,2)).astype(np.float32)) # agents,time,dim
        reward = torch.from_numpy(reward.astype(np.float32)) # scalar

        length = torch.from_numpy(np.array(length).astype(np.float32)) # scalar
        terminated = torch.from_numpy(terminated.astype(np.float32)) # time,1
        filled = torch.from_numpy(filled.astype(np.float32)) # time,agents
        
        if self.args.behavior: # transition:
            return observation,state,action,terminated,filled,length
        else:
            return observation,state,action,avail_action,reward,terminated,filled,length


