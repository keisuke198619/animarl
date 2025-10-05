from .. import MultiAgentEnv
import argparse
import gym
import numpy as np
from utils.utils_chase import *
import torch
from torch import nn, optim
from envs.animarl.chase_escape.network import DuelingNetwork
from pretrain.model.rnn_transition import RNNmodel 

class Animarl_Human_2vs1(MultiAgentEnv):
    def __init__(
        self,
        render=False,
        n_agents=2,
        time_limit=300,
        time_step=0,
        obs_dim=18, # 12
        env_name='animarl_human_2vs1',
        stacked=False,
        representation="simple",
        rewards='touch',
        logdir='animarl_human_dumps',
        write_video=False,
        transition=False,
        args=None,
        number_of_preys_controls=0,#1,
        number_of_preys=1,
        seed=0,
        action_dim=13,
        state_dim=12,
        space_dim=2,
        reward_dim=2, #3
        speed_pursuer=3.2, speed_evader=3, mass_pursuer1=1, mass_pursuer2=1,  mass_evader=1, 
        damping=0.25, dt=0.1, reward_share=False # (default) speed_pursuer=4.2/2.4/2.0, speed_evader=3.0/3.0/2.5
    ):
        self.render = render
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.obs_dim = obs_dim
        self.env_name = env_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_preys_controls = number_of_preys_controls
        self.seed = seed
        self.number_of_preys=number_of_preys
        # self.batch_size_run = 8
        
        self.n_all_agents = 3
        self.n_mate = self.n_all_agents -number_of_preys
        self.n_adv = number_of_preys

        self.speed_p1 = speed_pursuer
        self.speed_p2 = speed_pursuer
        self.speed_e = speed_evader
        self.mass_p1 = mass_pursuer1
        self.mass_p2 = mass_pursuer2
        self.mass_e = mass_evader
        self.damping = damping
        self.dt = dt
        self.reward_share = reward_share
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.space_dim = space_dim

        self.action_space = [gym.spaces.Discrete(
            action_dim) for _ in range(self.n_agents)]
            
        #self.observation_space = [
        #    gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        #]

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim  # QPLEX unit_dim for cds_gfootball
        # self.unit_dim = 8  # QPLEX unit_dim set like that in Starcraft II
        if n_agents == 2:
            self.device = torch.device('cpu')
            self.net_e = DuelingNetwork(18, 13).to(self.device)
            self.net_e.load_state_dict(torch.load("./envs/animarl/chase_escape/e_01.pth")) 
            """ Epsilon (not used) """ 
            epsilon_end = 0
            epsilon_begin = 1.0
            # epsilon_end = 0.1
            epsilon_decay = 10000
            self.n_step = 100000
            self.epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))
        
        self.transition = transition
        if self.transition:
            self.device = torch.device('cpu')
            # create argparse
            #parser = argparse.ArgumentParser()
            #parser.add_argument('--dummy', type=int, default=0, help='dummy')
            # args_env = parser.parse_args()
            args_env = args
            args_env.env = env_name
            args_env.n_all_agents = self.n_all_agents
            args_env.hidden_size = 64
            args_env.model_variant = 'RNN'
            args_env.n_layers = 1
            args_env.n_out = 2

            n_feature = obs_dim + action_dim
            try: 
                self.net_env = RNNmodel(n_feature, args_env).to(self.device)
                self.net_env.load_state_dict(torch.load("../AniMARL_results/model_transition/RNN_touch-animarl_human_2vs1/agent.th")) 
            except: import pdb; pdb.set_trace()
            

    def get_simple_obs(self, index=-1):

        if index == -1:
            simple_obs = self.state[None]         

        else:
            simple_obs = self.obs[index]
        
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def get_obs_e_KT(self, pos_e, vel_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp):
        
        pos_p1, vel_p1, pos_p2, vel_p2 = self.get_order_adv_KT(pos_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp)
        
        sub_pos_adv1 = get_sub_pos(pos_e, pos_p1)
        sub_pos_adv2 = get_sub_pos(pos_e, pos_p2)
        
        sub_vel_own_adv1 = get_sub_vel(pos_e, pos_p1, vel_e)
        sub_vel_own_adv2 = get_sub_vel(pos_e, pos_p2, vel_e)
            
        sub_vel_adv1 = get_sub_vel(pos_e, pos_p1, vel_p1)
        sub_vel_adv2 = get_sub_vel(pos_e, pos_p2, vel_p2)
                
        obs_e = np.concatenate([pos_e] + [sub_vel_own_adv1] + [sub_vel_own_adv2] + \
                            [pos_p1] + [sub_pos_adv1] + [sub_vel_adv1] + \
                            [pos_p2] + [sub_pos_adv2] + [sub_vel_adv2]).reshape(1,18)
        
        return obs_e   

    def get_order_adv_KT(self, pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp):
        dist1 = get_dist(pos_own, pos_adv1_tmp)
        dist2 = get_dist(pos_own, pos_adv2_tmp)

        d = [dist1, dist2]
        p = [pos_adv1_tmp, pos_adv2_tmp]
        v = [vel_adv1_tmp, vel_adv2_tmp]
        l = list(zip(d, p, v))
        l.sort()
        d, p, v = zip(*l)
        
        pos_adv1, vel_adv1 = p[0], v[0] 
        pos_adv2, vel_adv2 = p[1], v[1]
            
        return pos_adv1, vel_adv1, pos_adv2, vel_adv2

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1

        pos_p1,vel_p1,pos_p2,vel_p2,pos_e,vel_e = \
            self.state[:2],self.state[2:4],self.state[4:6],self.state[6:8],self.state[8:10],self.state[10:12]

        if self.n_agents == 3:
            action_p1, action_p2, action_e = actions.to('cpu').numpy().tolist()
        elif self.n_agents == 2:
            device = self.device
            action_p1, action_p2 = actions.to('cpu').numpy().tolist()
            obs_e = self.get_obs_e_KT(pos_e, vel_e, pos_p1, vel_p1, pos_p2, vel_p2)
            # feature_e = self.net_e.forward_com(obs_e.float().to(device))
            # feature0v_e = torch.matmul(feature_e, self.net_e.fc_state[0].weight.T) + self.net_e.fc_state[0].bias     
            # feature0a_e = torch.matmul(feature_e, self.net_e.fc_advantage[0].weight.T) + self.net_e.fc_advantage[0].bias
            # alue_e = self.net_e.forward(obs_e.float().to(device))
            action_e = self.net_e.act(torch.tensor(obs_e).float().to(device), -1) # deteministic # self.epsilon_func(self.n_step))

        if self.transition: 
            
            state_p1 = [pos_p1, vel_p1, pos_p2, vel_p2, pos_e, vel_e]
            state_p2 = [pos_p2, vel_p2, pos_p1, vel_p1, pos_e, vel_e]
            state_e = [pos_e, vel_e, pos_p1, vel_p1, pos_p2, vel_p2]
            obs_p1 = get_obs_p(state_p1,self.n_mate,self.n_adv)
            obs_p2 = get_obs_p(state_p2,self.n_mate,self.n_adv)
            obs_e = get_obs_e(state_e,self.n_mate,self.n_adv)
        
            obss = np.concatenate([obs_p1,obs_p2,obs_e],0)
            actions = np.array([action_p1,action_p2,action_e])
            actions = np.eye(self.n_actions)[actions]
            try: vel_out = self.net_env(torch.tensor(obss).float().to(self.device), torch.tensor(actions).float().to(self.device),device=self.device)
            except: import pdb; pdb.set_trace()
            vel_out = vel_out.detach().numpy()
            next_vel_p1,next_vel_p2, next_vel_e = vel_out[0,0,0,:], vel_out[0,0,1,:], vel_out[0,0,2,:]
            next_pos_p1 = pos_p1 + vel_p1*self.dt
            next_pos_p2 = pos_p2 + vel_p2*self.dt
            next_pos_e = pos_e + vel_e*self.dt
        else:
            abs_u_p1 = get_abs_u(action_p1, pos_p1, pos_e)
            next_pos_p1, next_vel_p1 = get_next_own_state(pos_p1, vel_p1, abs_u_p1, \
                                                        self.mass_p1, self.speed_p1, self.damping, self.dt) 

            abs_u_p2 = get_abs_u(action_p2, pos_p2, pos_e)
            next_pos_p2, next_vel_p2 = get_next_own_state(pos_p2, vel_p2, abs_u_p2, \
                                                        self.mass_p2, self.speed_p2, self.damping, self.dt) 
            state_adv = [pos_e, pos_p1, vel_p1, pos_p2, vel_p2]
            pos_adv1, _, _, _ = get_order_adv(state_adv,self.n_mate,self.n_adv)      
            abs_u_e = get_abs_u(action_e, pos_e, pos_adv1)
            next_pos_e, next_vel_e = get_next_own_state(pos_e, vel_e, abs_u_e, \
                                                        self.mass_e, self.speed_e, self.damping, self.dt)
        
        state_p1 = [next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2, next_pos_e, next_vel_e]
        state_p2 = [next_pos_p2, next_vel_p2, next_pos_p1, next_vel_p1, next_pos_e, next_vel_e]
        state_e = [next_pos_e, next_vel_e, next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2]
        next_obs_p1 = get_obs_p(state_p1,self.n_mate,self.n_adv)
        next_obs_p2 = get_obs_p(state_p2,self.n_mate,self.n_adv)
        next_obs_e = get_obs_e(state_e,self.n_mate,self.n_adv)
        #next_obs_p1 = self.get_obs_p(next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2, next_pos_e, next_vel_e)
        #next_obs_p2 = self.get_obs_p(next_pos_p2, next_vel_p2, next_pos_p1, next_vel_p1, next_pos_e, next_vel_e)
        #next_obs_e = self.get_obs_e(next_pos_e, next_vel_e, next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2)

        reward_p1 = self.get_reward_pursuer(next_pos_p1, next_pos_p2, next_pos_e, self.reward_share)
        reward_p2 = self.get_reward_pursuer(next_pos_p2, next_pos_p1, next_pos_e, self.reward_share)
        if self.n_agents == 3:
            reward_e = self.get_reward_evader(next_pos_e, next_pos_p1, next_pos_p2)
        # reward_p1 = get_reward_pursuer(next_pos_p1, next_pos_p2, next_pos_e, action_p1, self.reward_share)
        # reward_p2 = get_reward_pursuer(next_pos_p2, next_pos_p1, next_pos_e, action_p2, self.reward_share)
        # reward_e = get_reward_evader(next_pos_e, next_pos_p1, next_pos_p2, action_e)

        done = self.get_done(next_pos_e, next_pos_p1, next_pos_p2, self.time_step, self.episode_limit)
        
        self.state = np.concatenate([next_pos_p1,next_vel_p1,next_pos_p2,next_vel_p2,next_pos_e,next_vel_e],0)  
        self.obs = np.concatenate([next_obs_p1,next_obs_p2,next_obs_e],0) 
        if self.n_agents == 3:
            reward = np.array([reward_p1, reward_p2, reward_e]) 
        elif self.n_agents == 2:
            reward = np.array([reward_p1, reward_p2]) 

        if self.time_step >= self.episode_limit:
            infos = {'touch_reward': reward, "episode_limit": True}
        else:
            infos = {'touch_reward': reward}
        
        return reward, done, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.state_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        # if self.batch_size_run == 1:
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]
        #else:
        #    return [[[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)] for _ in range(self.batch_size_run)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self,initial=None):
        """Returns initial observations and states."""
        self.time_step = 0
        #self.env.reset() 
        #obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        if initial is not None:
            pos_p1 = initial[:2]
            vel_p1 = initial[2:4]
            pos_p2 = initial[4:6]
            vel_p2 = initial[6:8]
            pos_e = initial[8:10]
            vel_e = initial[10:12]

        else:
            pos_p1 = np.random.uniform(-0.5, 0.5, 2)
            vel_p1 = np.zeros(2)
            pos_p2 = np.random.uniform(-0.5, 0.5, 2)
            vel_p2 = np.zeros(2)
            pos_e = np.random.uniform(-0.5, 0.5, 2)
            vel_e = np.zeros(2)
        

        state_p1 = [pos_p1, vel_p1, pos_p2, vel_p2, pos_e, vel_e]
        state_p2 = [pos_p2, vel_p2, pos_p1, vel_p1, pos_e, vel_e]
        state_e = [pos_e, vel_e, pos_p1, vel_p1, pos_p2, vel_p2]

        obs = get_obs_p(state_p1,self.n_mate,self.n_adv) # obs_p1
        obs = np.concatenate([obs,get_obs_p(state_p2,self.n_mate,self.n_adv)],0) # obs_p2
        obs = np.concatenate([obs,get_obs_e(state_e,self.n_mate,self.n_adv)],0) # obs_e

        self.state = np.concatenate([pos_p1,vel_p1,pos_p2,vel_p2,pos_e,vel_e],0)  
        self.obs = obs
        return self.obs, self.state # 4*18,12

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    '''def get_obs_p(self, pos_p1, vel_p1, pos_p2, vel_p2, pos_e, vel_e):
    
        sub_pos_mate = get_sub_pos(pos_p1, pos_p2)
        sub_pos_adv = get_sub_pos(pos_p1, pos_e)

        obs_p = np.concatenate([pos_p1] + [vel_p1] + \
                            [sub_pos_mate] + [vel_p2] + \
                            [sub_pos_adv] + [vel_e]).reshape(1,12)

        return obs_p


    def get_obs_e(self, pos_e, vel_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp):
        
        pos_p1, vel_p1, pos_p2, vel_p2 = self.get_order_adv(pos_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp)
        
        sub_pos_adv1 = get_sub_pos(pos_e, pos_p1)
        sub_pos_adv2 = get_sub_pos(pos_e, pos_p2)

        obs_e = np.concatenate([pos_e] + [vel_e] + \
                            [sub_pos_adv1] + [vel_p1] + \
                            [sub_pos_adv2] + [vel_p2]).reshape(1,12)

        return obs_e   

    def get_order_adv(self, pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp):
        dist1 = get_dist(pos_own, pos_adv1_tmp)
        dist2 = get_dist(pos_own, pos_adv2_tmp)

        d = [dist1, dist2]
        p = [pos_adv1_tmp, pos_adv2_tmp]
        v = [vel_adv1_tmp, vel_adv2_tmp]
        l = list(zip(d, p, v))
        l.sort()
        d, p, v = zip(*l)
        
        pos_adv1, vel_adv1 = p[0], v[0] 
        pos_adv2, vel_adv2 = p[1], v[1]
            
        return pos_adv1, vel_adv1, pos_adv2, vel_adv2'''

    def get_reward_pursuer(self, abs_pos_own, abs_pos_mate, abs_pos_adv, reward_share):
        dist1 = get_dist(abs_pos_own, abs_pos_adv)
        dist2 = get_dist(abs_pos_mate, abs_pos_adv)
        reward = 0

        if reward_share == True:
            if dist1 < 0.1 or dist2 < 0.1:
                reward = 1
            elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
                reward = -10
        elif reward_share == False:
            if dist1 < 0.1:
                reward = 1
            elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
                reward = -10

        return reward


    def get_reward_evader(self, abs_pos_own, abs_pos_adv1, abs_pos_adv2):
        dist1 = get_dist(abs_pos_own, abs_pos_adv1)
        dist2 = get_dist(abs_pos_own, abs_pos_adv2)
        reward = 0

        if dist1 < 0.1 or dist2 < 0.1:
            reward = -1
        elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
            reward = -1

        return reward

    def get_done(self, abs_pos_own, abs_pos_adv1, abs_pos_adv2, num_step, max_step):
        dist1 = get_dist(abs_pos_own, abs_pos_adv1)
        dist2 = get_dist(abs_pos_own, abs_pos_adv2)

        if dist1 < 0.1 or dist2 < 0.1 or num_step > max_step or  \
        abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1 or \
        abs_pos_adv1[0] < -1 or abs_pos_adv1[1] < -1 or abs_pos_adv1[0] > 1 or abs_pos_adv1[1] > 1 or \
        abs_pos_adv2[0] < -1 or abs_pos_adv2[1] < -1 or abs_pos_adv2[0] > 1 or abs_pos_adv2[1] > 1:
            done = True
        else:
            done = False

        return done
