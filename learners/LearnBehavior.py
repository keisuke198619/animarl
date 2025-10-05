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

class LearnBehavior:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.env_name = self.args.env_args['env_name']
        self.space_dim = self.args.env_args['space_dim']

        if 'animarl' in self.env_name:
            self.params = [[] for n in range(args.n_agents)]
            
            for n in range(args.n_agents):
                self.params[n] = []
                self.params[n] += list(mac.agent.fc_in[n].parameters())
                try: self.params[n] += list(mac.agent.rnn[n].parameters())
                except: import pdb; pdb.set_trace()
                self.params[n] += list(mac.agent.fc_out[n].parameters())

        self.last_target_update_episode = 0
        

        # added
        # self.target_save = False

        self.mixer = None
        
        if 'animarl' in self.env_name and args.from_demo:
            lr = args.lr_adam_finetune if args.from_demo else args.lr_adam
        else: 
            lr = args.lr # args.lr_finetune if args.from_demo else 

        self.optimiser = [Adam(self.params[n], lr=lr) for n in range(args.n_agents)]
        self.args.gamma = self.args.gamma_animarl

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.list = [(np.arange(args.n_agents - i) + i).tolist() + np.arange(i).tolist()
                     for i in range(args.n_agents)]

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

        n_optim = self.args.n_agents
        criterion = th.nn.CrossEntropyLoss() # reduction='mean'
        
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

        mac_out, hidden_store, local_qs = self.mac.agent.forward(input_here.clone().detach(), initial_hidden_)

        #hidden_store = hidden_store.reshape(
        #    -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3) # 8, 151, 3, 64 (batch_size,time_limit,n_agents,h_size)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3) # 8,150,3 (batch_size,time_limit,n_agents)

        max_action_index = max_action_index.detach().unsqueeze(3) # (batch_size,time,n_agents,1)
        is_max_action = (max_action_index == actions).int().float()

        if use_demo:
            T_RL0 = mac_out.shape[1]-1
            T_demo0 = actions_demo.shape[1]
            action_out = x_mac_out[1::2, :T_RL0]
            n_agents = self.args.n_agents
            actions_demo_ = actions[::2] # RL

            actions_demo__,pos_RLs,pos_demos,T_RLs,T_demos,losses = [],[],[],[],[],[]
            for ba in range(int(batch.batch_size/2)):
                pos_RL = batch["state"][ba*2+1,:T_RL0,:n_agents*self.space_dim].reshape(T_RL0,n_agents,-1)[:,:,:self.space_dim].reshape(T_RL0,-1).detach().cpu().numpy()
                pos_demo = batch["state"][ba*2,:T_RL0,:n_agents*self.space_dim].reshape(T_RL0,n_agents,-1)[:,:,:self.space_dim].reshape(T_RL0,-1).detach().cpu().numpy()

                T_RL = T_RL0 if np.sum(pos_RL[:,0]==0) == 0 else np.where(pos_RL[:,0]==0)[0][0]
                T_demo = T_RL0 if np.sum(pos_demo[:,0]==0) == 0 else np.where(pos_demo[:,0]==0)[0][0]

                T = min(T_RL,T_demo)
                actions_demo__ = actions_demo_[ba,:T].clone().squeeze(2).long()
                actions_out_ = action_out[ba,:T].clone()
                loss_ = []
                for n in range(n_agents):
                    loss_.append(criterion(actions_out_[:,n], actions_demo__[:,n]))
                losses.append(th.stack(loss_))
            losses = th.stack(losses)

        loss,norm_loss_ = [],[]
        for paras in self.mac.agent.parameters():
            if paras.requires_grad:
                for para in paras:
                    norm_loss_.append(para.abs().sum())

        for n in range(n_optim):
            loss.append(self.args.lambda1 *losses[:,n].mean())
            # norm_loss_.append((norm_loss[:,:,n].clone() * mask_expand[:,:,n]).sum() / mask_expand[:,:,n].sum())
            norm_loss_[n] /= total_params
            loss[n] += self.args.lambda2 * norm_loss_[n]
        
        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        grad_norm = 0
        for n in range(n_optim):
            self.optimiser[n].zero_grad()
            loss[n].backward(retain_graph=True)
        # th.stack(loss).sum().backward()
        for n in range(n_optim):
            grad_norm += th.nn.utils.clip_grad_norm_(self.params[n], self.args.grad_norm_clip)
            self.optimiser[n].step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            #if n_optim == 1:
            #    self.logger.log_stat("loss", loss.item(), t_env)
            #else:
            self.logger.log_stat("loss", th.stack(loss).sum().item(), t_env)

            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            #self.logger.log_stat(
            #    "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            #self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
            #                     mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env

        return None # update_prior.squeeze().detach()

    def _update_targets(self):
        self.logger.console_logger.info("no target network")

    def cuda(self):
        for n in range(self.args.n_agents):
            self.mac.agent.fc_in[n].cuda()
            self.mac.agent.rnn[n].cuda()    
            self.mac.agent.fc_out[n].cuda()      
 

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