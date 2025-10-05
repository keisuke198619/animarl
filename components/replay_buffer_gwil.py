### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import ot
import scipy as sp
from dtaidistance.dtw_ndim import distance_fast
from dtaidistance.dtw_ndim import warping_paths

class ReplayBuffer_gwil(object):
    """Buffer to store environment transitions."""
    def __init__(self, args): # obs_shape, action_shape, capacity, device, cfg):
        capacity = args.t_max # capacity
        self.capacity = capacity
        self.device = args.device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 # if len(obs_shape) == 1 else np.uint8
        self.obs_dtype = obs_dtype
        obs_shape = args.obs_shape
        action_shape = args.action_dim
        n_agents = args.n_agents

        self.obses = np.empty((capacity, n_agents, obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, n_agents, obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, n_agents, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, n_agents, 1), dtype=np.float32)
        self.gw_rewards = np.empty((capacity, n_agents, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.idx_gw = 0
        self.full_gw = False

        self.args = args
        self.gw = args.gw
        self.dtw = args.dtw
        self.space_dim = args.env_args['space_dim']

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        try: 
            np.copyto(self.obses[self.idx], np.array(obs).astype(self.obs_dtype))
            np.copyto(self.actions[self.idx], np.array(action))
            np.copyto(self.rewards[self.idx], reward[:,None])
            np.copyto(self.next_obses[self.idx], np.array(next_obs).astype(self.obs_dtype))
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        except: import pdb; pdb.set_trace()

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def process_trajectory(self, traj_expert, metric_expert = 'euclidean', metric_agent = 'euclidean', sinkhorn_reg=5e-3, normalize_agent_with_expert=False, include_actions=True, entropic=True):
        assert not (self.idx == 0 and not self.full)
        if self.idx == 0:
            traj_agent = self.obses[self.idx_gw:]
        elif self.idx_gw > self.idx:
            traj_agent = np.concatenate([self.obses[self.idx_gw:],self.obses[:self.idx]],0)
            print("self.idx_gw > self.idx: "+str(self.idx_gw > self.idx))
        else:
            traj_agent = self.obses[self.idx_gw:self.idx]

        if normalize_agent_with_expert:
            traj_agent = (traj_agent - traj_expert.mean())/(traj_expert.std())
        
        if include_actions:
            if self.idx == 0:
                actions_trajectory = self.actions[self.idx_gw:]
            elif self.idx_gw > self.idx:
                print('TBD when self.idx_gw > self.idx')
                import pdb; pdb.set_trace()
            else:
                actions_trajectory = self.actions[self.idx_gw:self.idx]
            traj_agent = np.concatenate((traj_agent,actions_trajectory), axis=1)

        gw_rewards = self.compute_gw_reward(traj_expert, traj_agent, metric_expert, metric_agent,
                                                          entropic, sinkhorn_reg=sinkhorn_reg)

        if self.idx == 0:
            self.gw_rewards[self.idx_gw:] = np.expand_dims(gw_rewards, axis=1)
            normalized_reward = ((self.gw_rewards[:self.idx] - self.gw_rewards[:self.idx].mean())/(1e-5+self.gw_rewards[:self.idx].std()))[self.idx_gw:].sum()
        elif self.idx_gw > self.idx:
            self.gw_rewards[self.idx_gw:] = np.expand_dims(gw_rewards[:-self.idx], axis=2)
            self.gw_rewards[:self.idx] = np.expand_dims(gw_rewards[-self.idx:], axis=2)
            normalized_reward = ((self.gw_rewards - self.gw_rewards.mean())/(1e-5+self.gw_rewards.std()))
            normalized_reward = np.concatenate([normalized_reward[self.idx_gw:],normalized_reward[:self.idx]],0).sum()
        else:
            self.gw_rewards[self.idx_gw:self.idx] = np.expand_dims(gw_rewards, axis=2)
            normalized_reward = ((self.gw_rewards[:self.idx] - self.gw_rewards[:self.idx].mean())/(1e-5+self.gw_rewards[:self.idx].std()))[self.idx_gw:self.idx].sum()

        self.idx_gw = self.idx

        return gw_rewards.sum(0), normalized_reward

    def compute_gw_reward(self, traj_expert, traj_agent, metric_expert = 'euclidean', metric_agent = 'euclidean', entropic=True, sinkhorn_reg=5e-3, return_coupling = False):
        distances_experts, distances_agents,optimal_couplings,rewards = [], [], [], []
        for n in range(self.args.n_agents):
            try: distances_expert = sp.spatial.distance.cdist(traj_expert[n], traj_expert[n], metric=metric_expert)
            except: import pdb; pdb.set_trace()

            distances_agent = sp.spatial.distance.cdist(traj_agent[:,n], traj_agent[:,n], metric=metric_agent)
            distances_experts.append(distances_expert)
            distances_agents.append(distances_agent)

        distances_expert = np.array(distances_experts)
        distances_agent = np.array(distances_agents)
        
        distances_expert/=distances_expert.max()+1e-6
        try: distances_agent/=distances_agent.max()+1e-6
        except: import pdb; pdb.set_trace()

        distribution_expert = ot.unif(traj_expert.shape[1])
        distribution_agent = ot.unif(traj_agent.shape[0])

        for n in range(self.args.n_agents):
            if self.gw:
                # optimal_coupling, constC, tens: T1 x T2
                # distances_expert, hExpert: T1 x T1
                # distances_agent, hAgent: T2 x T2
                # rewards: T2
                if entropic:
                    optimal_coupling = ot.gromov.entropic_gromov_wasserstein(
                        distances_expert[n], distances_agent[n], distribution_expert, distribution_agent, 'square_loss', epsilon=sinkhorn_reg, max_iter=1000, tol=1e-9)
                else:
                    optimal_coupling= ot.gromov.gromov_wasserstein(distances_expert[n], distances_agent[n], distribution_expert, distribution_agent, 'square_loss')
            

                constC, hExpert, hAgent = ot.gromov.init_matrix(distances_expert[n], distances_agent[n], distribution_expert, distribution_agent, loss_fun='square_loss')

                tens = ot.gromov.tensor_product(constC, hExpert, hAgent, optimal_coupling)

                reward = -(tens*optimal_coupling).sum(axis=0)

                optimal_couplings.append(optimal_coupling)
            elif self.dtw:
                try: 
                    dtw, distance_matrix_ = warping_paths(traj_agent[:,n,:self.space_dim].astype(np.float64), traj_expert[n,:,:self.space_dim].astype(np.float64))
                    reward = (distance_matrix_[1:,1:]).sum(axis=1)
                    # reward = -distance_fast(traj_agent[:,n,:self.space_dim].astype(np.float64), traj_expert[n,:,:self.space_dim].astype(np.float64))
                except: import pdb; pdb.set_trace()

            rewards.append(reward)

        if self.gw:
            optimal_couplings = np.array(optimal_couplings).transpose()
        rewards = np.array(rewards).transpose()

        if return_coupling:
            return rewards, optimal_coupling

        return rewards

    def sample(self, batch_size, gw=False, normalize_reward=False,normalize_reward_batch=False, include_external_reward=False, weight_external_reward=1, weight_gw_reward=1):

        if gw or self.dtw:
            end_idxs = self.capacity if self.full_gw else self.idx_gw
        else:
            end_idxs = self.capacity if self.full else self.idx

        try: idxs = np.random.randint(0,end_idxs,size=batch_size)
        except: import pdb; pdb.set_trace()

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        if gw or self.dtw:
            if normalize_reward_batch:
                rewards = torch.as_tensor((self.gw_rewards[idxs] - self.gw_rewards[idxs].mean())/(1e-5+self.gw_rewards[idxs].std()), device=self.device)
            elif normalize_reward:
                gw_rewards_normalized = (self.gw_rewards[:end_idxs] - self.gw_rewards[:end_idxs].mean())/(1e-5+self.gw_rewards[:end_idxs].std())
                rewards = torch.as_tensor(gw_rewards_normalized[idxs], device=self.device)
            else:
                rewards = torch.as_tensor(self.gw_rewards[idxs], device=self.device)

        else:
            rewards = torch.as_tensor(self.rewards[idxs], device=self.device)

        if include_external_reward:
            # assert gw or self.dtw
            if gw or self.dtw:
                rewards=weight_gw_reward*rewards+weight_external_reward*torch.as_tensor(self.rewards[idxs], device=self.device)
            else:
                rewards=torch.as_tensor(self.rewards[idxs], device=self.device)

        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
