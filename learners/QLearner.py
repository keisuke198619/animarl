import copy
import torch as th
import numpy as np
import torch.nn.functional as F

from torch.optim import RMSprop, Adam
from components.episode_buffer import EpisodeBatch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from modules.CDS.predict_net import Predict_Network, Predict_Network_WithID, Predict_ID_obs_tau
from dtaidistance.dtw_ndim import warping_paths
from utils.utils_chase import *

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.env_name = self.args.env_args['env_name']
        self.space_dim = self.args.env_args['space_dim']

        self.last_target_update_episode = 0
        
        # added
        # self.target_save = False

        self.mixer = None
        
        if 'animarl' in self.env_name and args.from_demo:
            lr = args.lr_adam_finetune if args.from_demo else args.lr_adam
        else: 
            lr = args.lr # args.lr_finetune if args.from_demo else 

        self.optimiser = [Adam(self.mac.agent.params[n], lr=lr) for n in range(args.n_agents)]
        self.args.gamma = self.args.gamma_animarl

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        if self.args.double_q: # added 
            self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.list = [(np.arange(args.n_agents - i) + i).tolist() + np.arange(i).tolist()
                     for i in range(args.n_agents)]

        self.supervisedLoss = [th.nn.NLLLoss(ignore_index=-100) for _ in range(args.n_agents)]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, use_demo=False, actions_demo=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] # 8,150,1 (batch_size,time_limit,1)
        actions = batch["actions"][:, :-1] # 8,150,3,1 (batch_size,time_limit,n_agents,1)
        terminated = batch["terminated"][:, :-1].float() # 8,150,1 (batch_size,time_limit,1)
        mask = batch["filled"][:, :-1].float() # 8,150,1 (batch_size,time_limit,1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"] # 8,151,3,19 (batch_size,time_limit,n_agents,n_actions)
        actions_onehot = batch["actions_onehot"][:, :-1] # 8,151,3,19 (batch_size,time_limit,n_agents,n_actions)
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions
        
        if self.args.alg_DIL:
            no_direct_reward = True
        else:
            no_direct_reward = False
        if no_direct_reward:
            rewards *= 0 # 1e-10

        n_optim = self.args.n_agents
        
        total_params = 0
        for param in self.mac.agent.parameters():
            total_params += param.numel()/self.args.n_agents

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)
        if self.mac.hidden_states is not None:
            initial_hidden = self.mac.hidden_states.clone().detach()
            initial_hidden = initial_hidden.reshape(
                -1, initial_hidden.shape[-1]).to(self.args.device)
            initial_hidden_ = initial_hidden.clone().detach()
        else:
            initial_hidden_ = None
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device) # 8, 3, 151, 45 (batch_size,n_agents,time_limit,n_input)

        mac_out, hidden_store, out_cond = self.mac.agent.forward(input_here.clone().detach(), initial_hidden_)

        #hidden_store = hidden_store.reshape(
        #    -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3) # 8, 151, 3, 64 (batch_size,time_limit,n_agents,h_size)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3) # 8,150,3 (batch_size,time_limit,n_agents)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if use_demo:
            T_RL0 = mac_out.shape[1]-1
            T_demo0 = actions_demo.shape[1]
            A = max_action_index[1::2,:T_RL0]
            n_agents = self.args.n_agents
            actions_demo_ = actions[::2] # RL

            if self.args.cond == 'CF' or self.args.cond == 'CF2':                
                # https://github.com/keisuke198619/TGV-CRN/blob/main/train_MADSW.py
                loss_cond = [torch.zeros(1).to(self.args.device) for _ in range(n_agents)]
                for ba in range(int(batch.batch_size/2)):
                    pos_RL = batch["state"][ba*2+1,:T_RL0,:n_agents*self.space_dim].reshape(T_RL0,n_agents,-1)[:,:,:self.space_dim].reshape(T_RL0,-1).detach().cpu().numpy()
                    T_RL = T_RL0 if np.sum(pos_RL[:,0]==0) == 0 else np.where(pos_RL[:,0]==0)[0][0]
                    for n in range(n_agents):
                        cond_pred = out_cond[ba,:T_RL,n,:]
                        cond_gt = batch["state"][ba*2,0,-1:].long()
                        
                        loss_cond[n] += F.cross_entropy(cond_pred, cond_gt.expand(T_RL, 1).reshape(-1)).sum()/T_RL
                for n in range(n_agents):
                    loss_cond[n] /= batch.batch_size/2

            # DIL and AAS
            actions_demo__,pos_RLs,pos_demos,T_RLs,T_demos = [],[],[],[],[]
            for ba in range(int(batch.batch_size/2)):
                pos_RL = batch["state"][ba*2+1,:T_RL0,:n_agents*self.space_dim].reshape(T_RL0,n_agents,-1)[:,:,:self.space_dim].reshape(T_RL0,-1).detach().cpu().numpy()
                pos_demo = batch["state"][ba*2,:T_RL0,:n_agents*self.space_dim].reshape(T_RL0,n_agents,-1)[:,:,:self.space_dim].reshape(T_RL0,-1).detach().cpu().numpy()
                T_RL = T_RL0 if np.sum(pos_RL[:,0]==0) == 0 else np.where(pos_RL[:,0]==0)[0][0]
                T_demo = T_RL0 if np.sum(pos_demo[:,0]==0) == 0 else np.where(pos_demo[:,0]==0)[0][0]
                rewards[ba*2,:T_RL-1,:] = 0 # demo

                if self.args.alg_DIL:
                    for n in range(n_agents):   
                        traj_expert = pos_demo[:T_demo,n*self.space_dim:(n+1)*self.space_dim]
                        traj_agent = pos_RL[:T_RL,n*self.space_dim:(n+1)*self.space_dim]
                        dtw, distance_matrix_ = warping_paths(traj_agent, traj_expert) 
                        reward = torch.tensor(-(distance_matrix_[1:,1:]).min(axis=1)).to(self.args.device)
                        rewards[ba*2+1,:T_RL,n] += self.args.lambda3 * reward
                        if "silkmoth" in self.env_name:
                            rewards[ba*2+1,:T_RL,n] += - np.abs(T_RL-T_demo)/600
                    
                elif self.args.alg_AAS:
                    # T_RL >= T_demo:
                    dtw, distance_matrix_ = warping_paths(pos_demo[:T_demo,:],pos_RL[:T_RL,:]) # ,psi=(0, T_demo, 0, T_RL))
                    index_dist = np.argmin(distance_matrix_[1:,1:],axis=0) 
                    actions_demo_ba = actions_demo_[ba].clone()
                    actions_demo_ba[:T_RL] = actions_demo_[ba,index_dist]
                    actions_demo__.append(actions_demo_ba)

            if self.args.alg_AAS:        
                actions_demo_ = th.stack(actions_demo__,0)


        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals -
                      chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        if self.args.double_q: # added 
            self.target_mac.init_hidden(batch.batch_size)
            if self.target_mac.hidden_states is not None:
                initial_hidden_target = self.target_mac.hidden_states.clone().detach()
                initial_hidden_target = initial_hidden_target.reshape(
                    -1, initial_hidden_target.shape[-1]).to(self.args.device)
                initial_hidden_target_ = initial_hidden_target.clone().detach()
            else:
                initial_hidden_target_ = None
            target_mac_out, _, _ = self.target_mac.agent.forward(
                input_here.clone().detach(), initial_hidden_target_)
            target_mac_out = target_mac_out[:, 1:] # 8,150,3,19 (batch_size,time_limit,n_agents,n_actions)

        # Max over target Q-Values
        if self.args.double_q: # default
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
        # else: # commented
        #    target_max_qvals = target_mac_out.max(dim=3)[0]


        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals # self.args.beta * intrinsic_rewards + \

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()

            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # take mean over actual data                               
        update_prior = (masked_td_error ** 2).squeeze().sum(dim=-1,
                                                            keepdim=True) / mask.squeeze().sum(dim=-1, keepdim=True)
        
        loss,norm_loss_ = [],[]
        try: 
            for paras in self.mac.agent.parameters():
                if paras.requires_grad:
                    for para in paras:
                        try: norm_loss_.append(para.abs().sum())
                        except: import pdb; pdb.set_trace()     
        except: import pdb; pdb.set_trace()     
        for n in range(n_optim):
            loss.append(self.args.lambda1 * (masked_td_error[:,:,n].clone() ** 2).sum() / mask[:,:,n].sum())
            # norm_loss_.append((norm_loss[:,:,n].clone() * mask_expand[:,:,n]).sum() / mask_expand[:,:,n].sum())
            norm_loss_[n] /= total_params
            loss[n] += self.args.lambda2 * norm_loss_[n]
        
        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        if use_demo: # 
            if  self.args.cond == 'CF' or self.args.cond == 'CF2':
                for n in range(n_optim):
                    try: loss[n] += self.args.lambda4 * loss_cond[n].squeeze()
                    except: import pdb; pdb.set_trace()

            if self.args.alg_AAS:
                logsoftmax = F.log_softmax(mac_out[::2, :-1].clone(),dim=3)
                n_actions = mac_out.shape[-1]
                actions_demo_masked = actions_demo_.clone() 
                mask_expand_ = mask[::2].unsqueeze(-1).expand_as(actions_demo_masked)

                for n in range(n_optim):
                    loss[n] += self.args.lambda3 * self.supervisedLoss[n](logsoftmax[:,:,n].reshape(-1,n_actions),actions_demo_masked[:,:,n].reshape(-1,))

        # Optimise
        grad_norm = 0
        for n in range(n_optim):
            self.optimiser[n].zero_grad()
            try: loss[n].backward(retain_graph=True)
            except: import pdb; pdb.set_trace()
        # th.stack(loss).sum().backward()
        for n in range(n_optim):
            grad_norm += th.nn.utils.clip_grad_norm_(self.mac.agent.params[n], self.args.grad_norm_clip)
            self.optimiser[n].step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            #if n_optim == 1:
            #    try: self.logger.log_stat("loss", loss.item(), t_env)
            #    except: import pdb; pdb.set_trace()
            #else:
            self.logger.log_stat("loss", th.stack(loss).sum().item(), t_env)

            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                 mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(
                "target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env

        return update_prior.squeeze().detach()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        for n in range(self.args.n_agents):
            self.mac.agent.fc_in[n].cuda()
            self.target_mac.agent.fc_in[n].cuda()
            if self.args.agent == 'rnn':
                self.mac.agent.rnn[n].cuda()
                self.target_mac.agent.rnn[n].cuda()
            if self.args.dueling:
                self.mac.agent.fc_state[n].cuda()
                self.mac.agent.fc_advantage[n].cuda()
                self.target_mac.agent.fc_state[n].cuda()
                self.target_mac.agent.fc_advantage[n].cuda()
            else:    
                self.mac.agent.fc_out[n].cuda()      
                self.target_mac.agent.fc_out[n].cuda()
            if self.args.cond == 'CF' or self.args.cond == 'CF2':
                self.mac.agent.fc_out_cond[n].cuda()
                self.target_mac.agent.fc_out_cond[n].cuda()
 
    def save_models(self, path):
        
        self.mac.save_models(path)
        for n in range(self.args.n_agents):
            th.save(self.optimiser[n].state_dict(), "{}/opt_{}.th".format(path,str(n)))

    def load_models(self, path, pretrain=False, mixer=False, opt=True):
        self.mac.load_models(path)
        if opt:
            for n in range(self.args.n_agents):
                self.optimiser[n].load_state_dict(
                    th.load("{}/opt_{}.th".format(path,str(n)), map_location=lambda storage, loc: storage))