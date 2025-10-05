from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import csv
import os, copy
import torch as th
from utils.utils_chase import *
# import dtaidistance # dtw, fastdtw
from dtaidistance.dtw_ndim import distance_fast

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # assert self.batch_size == 1
        self.args.env_args['args'] = copy.deepcopy(args)
        self.burn_in = args.burn_in
        # if args.seed >= 11 and 'animarl_agent' in self.args.env_args['env_name']:
        #    self.args.env_args['n_agents'] += 1
        #    self.args.env_args['reward_dim'] += 1
        # if args.cond == 'CF' and "animarl_agent_2vs1" in args.env:
        #    env_info["episode_limit"] = 148
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if args.multi_env:
            env_args_ = self.args.env_args.copy()
            if 'animarl_agent' in self.args.env_args['env_name']:
                # self.n_demo = np.arange(400,460) # (100) # [0,1,2,3,4,5,6,7,8,9]
                all_indices = np.arange(800)
                self.n_demo = np.concatenate([all_indices[200:225],all_indices[579:604],all_indices[286:293],all_indices[294:312],all_indices[663:670],all_indices[671:689]]) # ,all_indices[286:291],all_indices[663:668]])
            elif 'animarl_silkmoth' in self.args.env_args['env_name']:
                ind_ = np.arange(60)
                self.n_demo = np.concatenate([ind_[:6],ind_[30:36],ind_[6:12],ind_[36:42],ind_[12:18],ind_[42:48],ind_[18:21],ind_[22:30],ind_[48:60]],0) # ind_[18:24],ind_[48:54],ind_[24:30],ind_[54:60]],0) # movie 1-4/5
                '''self.n_demo = np.arange(120) # 240) 
                n_test = int(np.floor(args.seed/10))-1
                if n_test == 0:
                    self.n_demo = np.concatenate([self.n_demo[(n_test+1)*30:],self.n_demo[:(n_test+1)*30]])
                else:
                    self.n_demo = np.concatenate([self.n_demo[:n_test*30],self.n_demo[(n_test+1)*30:],self.n_demo[n_test*30:(n_test+1)*30]])
                '''                
            elif 'animarl_fly' in self.args.env_args['env_name']:
                self.n_demo = np.arange(37,107) # (47,107) # (50,114) 
            elif 'animarl_newt' in self.args.env_args['env_name']:
                self.n_demo = np.arange(210,280)#270) 
            else:
                import pdb; pdb.set_trace()
            self.env_demo = [[] for _ in range(len(self.n_demo))] # 13
            n = 0
            for n_demo in self.n_demo:
                env_args_['env_name'] = self.args.env_args['env_name']+'_'+str(n_demo)
                self.env_demo[n] = env_REGISTRY[self.args.env_args['env_name']](**env_args_)
                n += 1
                self.episode_limit = self.env_demo[0].episode_limit
            '''self.episode_limit = self.env.episode_limit'''
        else:
            self.episode_limit = self.env.episode_limit
        self.t = 0
    
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.train_dist = []
        self.test_dist = []

        # Log the first run
        self.env_name = self.args.env_args['env_name']
        self.log_train_stats_t = -1000000
        env_name = args.env_args['env_name']
        self.csv_dir = f'./csv_files/{env_name}/{args.name}'   
        # self.csv_dir = f'./csv_files/{args.name}'

        if args.alg_RNN:
            self.csv_dir += '_RNN'
        if args.pretrain: 
            self.csv_dir += '_pretrain'
        if args.from_demo: 
            self.csv_dir += '_from_demo'
        if args.alg_AAS: 
            self.csv_dir += '_AAS'
        if args.alg_DIL:
            self.csv_dir += '_DIL'
        if args.alg_BC:
            self.csv_dir += '_BC'
        
        self.csv_dir += args.variant

        self.csv_dir += '_' + args.cond 
        #if args.cont != 'cont' and args.cont != False:
        #    self.csv_dir = f'./csv_files/{args.name[:-6]}_{args.cont}/{env_name}/'
        #else:
        #    self.csv_dir = f'./csv_files/{args.name[:-6]}/{env_name}/'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}.csv'
        print('csv_path: '+self.csv_path)
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir, exist_ok=True)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self,initial=None):
        self.batch = self.new_batch()
        self.env.reset() if initial is None else self.env.reset(initial)
        self.t = 0

    def writereward(self, win_rate, dist, step):
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, win_rate, dist])
        else:
            with open(self.csv_path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'win_rate', 'distance'])
                csv_write.writerow([step, win_rate, dist])

    def run(self, n=0, states_demo=None,test_mode=False, action_demo=None, buffer=None, info=None, save=False):
        if states_demo is not None:
            self.env = self.env_demo[n]
            try: initials = states_demo[0]
            except: import pdb; pdb.set_trace()
        if states_demo is None:
            self.reset() # why does the reset create video when n=1?
        else:
            initials = states_demo[0]
            self.reset(initials) 

        terminated = False
        episode_return = 0.0
        self.mac.init_hidden(batch_size=self.batch_size)

        if self.args.mac == "oda_mac":
            self.mac.init_latent(batch_size=self.batch_size)

        states = []
        if save:
            avail_actions,obs,actionss,rewards,ts,agent_outputss = [],[],[],[],[],[]
        else:
            rewards,actionss = [],[]
        
        while not terminated:
            # print("runner"+str(self.t))
            #if self.t == 1:
            #    self.env.env.unwrapped.observation()[0]['left_team']
            #    import pdb; pdb.set_trace()

            # if initials is not None:
            #     # state__ = self.env_demo[n].env.unwrapped.get_state()
            #     # self.env.env.unwrapped.set_state(state__) # cannnot do this
            
            if self.t >= self.burn_in:
                state_ = self.env.get_state() #if initials is None or self.t > 0 else self.env.get_state(initials)
                obs_ = self.env.get_obs() # if initials is None or self.t > 0 else self.env.get_obs(initials)
                states_demo_ = None
            else:
                states_demo_ = states_demo
                state_ = states_demo[self.t,None]
                if 'animarl_newt' in self.env_name: # or 'animarl_agent' in self.env_name or 'animarl_fly' in self.env_name:
                    obs_ = self.env.get_obs() # [agents][feat]
                    state_p1 = [state_[0,:2],state_[0,2:4],state_[0,4:6],state_[0,6:8],state_[0,8:10],state_[0,10:]]  # [pos_p1, vel_p1, pos_p2, vel_p2, pos_e, vel_e]
                    state_p2 = [state_[0,4:6],state_[0,6:8],state_[0,:2],state_[0,2:4],state_[0,8:10],state_[0,10:]] # [pos_p2, vel_p2, pos_p1, vel_p1, pos_e, vel_e]
                    state_e = [state_[0,8:10],state_[0,10:],state_[0,:2],state_[0,2:4],state_[0,4:6],state_[0,6:8]] # [pos_e, vel_e, pos_p1, vel_p1, pos_p2, vel_p2]

                    obs_ = get_obs_p(state_p1,self.env.n_mate,self.env.n_adv) # obs_p1
                    obs_ = np.concatenate([obs_,get_obs_p(state_p2,self.env.n_mate,self.env.n_adv)],0) # obs_p2
                    obs_ = np.concatenate([obs_,get_obs_e(state_e,self.env.n_mate,self.env.n_adv)],0) # obs_e

                elif 'silkmoth' in self.env_name:
                    obs_ = state_[:,2:] # remove position
                elif 'bat' in self.env_name or 'dragonfly' in self.env_name:
                    import pdb; pdb.set_trace()
                else:
                    import pdb; pdb.set_trace()
            
            avail_action_ = self.env.get_avail_actions()
            states.append(state_.copy())
            #if self.env_name == "animarl_agent_2vs1": # or self.env_name == "animarl_silkmoth":
            #    obs_ = [np.concatenate([obs_[0],state_[0,-1:]]) for n in range(self.args.n_agents)]

            if save:
                avail_actions.append(np.array(avail_action_).copy())
                obs.append(np.array(obs_).copy())
                # actionss.append(actions.to('cpu').detach().numpy().copy())

            pre_transition_data = {
                "state": [state_], # [1,feat]
                "avail_actions": [avail_action_], # [agents][actions]
                "obs": [obs_] # [agents,feat]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.t >= self.burn_in: # action_demo is None:
                actions, agent_outputs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                if self.args.env == "animarl_silkmoth":
                    reward, terminated, env_info = self.env.step(actions[0], info)
                else: 
                    reward, terminated, env_info = self.env.step(actions[0]) # ,states_demo_
            else:
                # actions = th.tensor(action_demo[self.t,None]).to('cuda')
                actions = th.argmax(th.tensor(action_demo[self.t, None]), dim=-1).to('cuda')
                if self.args.env == "animarl_silkmoth":
                    reward, terminated, env_info = self.env.step(actions[0], info)
                else:
                    reward, terminated, env_info = self.env.step(actions[0])

            episode_return += reward
            action = actions.to('cpu').detach().numpy().copy()
            post_transition_data = {
                    "actions": actions, # [1,agents]
                    "reward": [(reward,)], # int
                    "terminated": [(terminated != env_info.get("episode_limit", False),)], # bool
                }
            self.batch.update(post_transition_data, ts=self.t)

            if save:
                # states.append(self.env.get_state().copy())
                # avail_actions.append(np.array(self.env.get_avail_actions()).copy())
                # obs.append(np.array(self.env.get_obs()).copy())
                actionss.append(action)
                if self.t >= self.burn_in:
                    agent_outputss.append(agent_outputs.to('cpu').detach().numpy().copy())
                rewards.append(np.array((reward,)))
                ts.append(np.array((self.t,)))
            else:
                actionss.append(action)
                rewards.append(np.array((reward,)))
            #if n==7:
            #    import pdb; pdb.set_trace()
            self.t += 1

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                         for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # compute DTW
        states = np.array(states)
        
        if 'animarl' in self.args.env_args['env_name']:
            pos_RL = states[:,0,:self.args.n_agents*2].reshape((states.shape[0],self.args.n_agents,-1))[:,:,:self.env.space_dim].reshape((states.shape[0],-1))
            pos_demo = states_demo[:,:self.args.n_agents*2].reshape((states_demo.shape[0],self.args.n_agents,-1))[:,:,:self.env.space_dim].reshape((states_demo.shape[0],-1))
                
        else:
            print('not defined')
            import pdb; pdb.set_trace()
        
        cur_dist = self.test_dist if test_mode else self.train_dist
        # cur_stats["dtw"] = distance_fast(pos_RL[:,:1], pos_demo[:,:1])
        cur_dist_ = distance_fast(pos_RL.astype('float64'), pos_demo.astype('float64'))
        cur_dist.append(cur_dist_) # [:,:1]
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            '''if np.isscalar(cur_returns[0]) or (th.is_tensor(cur_returns) and np.isscalar(cur_returns[0].to('cpu').tolist())): 
                cur_returns_mean = np.array(
                    [0 if item <= 0 else 1 for item in cur_returns]).mean()
            else:'''
            cur_returns_ = np.array(cur_returns)
            cur_returns_[cur_returns_<=0] = 0
            # cur_returns_[cur_returns_>0] = 1
            cur_returns_mean = cur_returns_.mean(0)
                
            dist = np.array(cur_dist).mean()
            # if not save:
            self.writereward(cur_returns_mean, dist, self.t_env)

        print(str(np.array(rewards).sum(0))+', t = '+str(self.t)+' ep_length: '+str(cur_stats.get("ep_length", 0)))

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, cur_dist, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, cur_dist, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if save:
            avail_actions,obs,actionss = np.array(avail_actions),np.array(obs),np.array(actionss)
            rewards,ts = np.array(rewards),np.array(ts) 

            return states,avail_actions,obs,actionss,rewards,ts,agent_outputss,cur_dist_
        else:
            return self.batch

    def _log(self, returns, stats, dist, prefix):
        self.logger.log_stat(prefix + "return_mean",
                             np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "dist_mean",
                             np.mean(dist), self.t_env)
        self.logger.log_stat(prefix + "dist_std",
                             np.std(dist), self.t_env)
        dist.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        
        stats.clear()
