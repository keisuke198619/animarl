from .. import MultiAgentEnv
import argparse
import gym
import numpy as np
from utils.utils_chase import *
import torch
from torch import nn, optim
from modules.agents.mlp_agent import MLPAgent

class Animarl_Newt_2vs1(MultiAgentEnv):
    def __init__(
        self,
        n_agents=2,
        time_limit=300,
        time_step=0,
        obs_dim=18, # 12
        env_name='animarl_newt_2vs1',
        stacked=False,
        representation="simple",
        rewards='touch',
        logdir='animarl_flies_dumps',
        transition=False,
        args=None,
        number_of_opponents_controls=0,#1,
        number_of_opponents=1,
        seed=0,
        action_dim=13,
        state_dim=12,
        space_dim=2,
        reward_dim=2, #3
        dt=0.1,
        reward_share=False
    ):
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.obs_dim = obs_dim
        self.env_name = env_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.number_of_opponents_controls = number_of_opponents_controls
        self.seed = seed
        self.number_of_opponents=number_of_opponents
        # self.batch_size_run = 8
        
        self.n_all_agents = 3
        self.n_mate = self.n_all_agents -number_of_opponents
        self.n_adv = number_of_opponents
        self.speed_p1 = args.params["speed_p1"] # speed_pursuer
        self.speed_p2 = args.params["speed_p2"] # speed_pursuer
        self.speed_e = args.params["speed_e"] # speed_evader
        self.mass_p1 = args.params["mass_p1"] # mass_pursuer1
        self.mass_p2 = args.params["mass_p2"] # mass_pursuer2
        self.mass_e = args.params["mass_e"] # mass_evader
        # self.damping = damping
        self.damping_p1 = args.params["damping_p1"]
        self.damping_p2 = args.params["damping_p2"]
        self.damping_e = args.params["damping_e"]
        self.dt = dt
        self.reward_share = reward_share
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.space_dim = space_dim

        self.action_space = [gym.spaces.Discrete(
            action_dim) for _ in range(self.n_agents)]

        self.n_actions = self.action_space[0].n
        self.unit_dim = self.obs_dim  # QPLEX unit_dim for cds_gfootball
        self.reward_out = -5       

    def get_simple_obs(self, index=-1):

        if index == -1:
            simple_obs = self.state[None]         

        else:
            simple_obs = self.obs[index]
        
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)


    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1

        action_p1, action_p2, action_e = actions.detach().to('cpu').numpy().tolist()
    
        params = {
            'mass_p1': self.mass_p1,
            'speed_p1': self.speed_p1,
            'damping_p1': self.damping_p1,
            'dt': self.dt,
            'mass_p2': self.mass_p2,
            'speed_p2': self.speed_p2,
            'damping_p2': self.damping_p2,
            'mass_e': self.mass_e,
            'speed_e': self.speed_e,
            'damping_e': self.damping_e,
            'n_mate': self.n_mate,
            'n_adv': self.n_adv
        }

        state_p1, state_p2, state_e, next_pos_p1, next_pos_p2, next_pos_e = transition_agent(self.state, action_p1, action_p2, action_e, params, boundary='square')

        self.state = state_p1.reshape((-1,))
        
        next_obs_p1 = get_obs_p(state_p1,self.n_mate,self.n_adv)
        next_obs_p2 = get_obs_p(state_p2,self.n_mate,self.n_adv)
        next_obs_e = get_obs_e(state_e,self.n_mate,self.n_adv)

        reward_p1 = self.get_reward_pursuer(next_pos_p1, next_pos_p2, next_pos_e, self.reward_share)
        reward_p2 = self.get_reward_pursuer(next_pos_p2, next_pos_p1, next_pos_e, self.reward_share)
        reward_e = self.get_reward_evader(next_pos_e, next_pos_p1, next_pos_p2)

        done = self.get_done(next_pos_e, next_pos_p1, next_pos_p2, self.time_step, self.episode_limit)
        if done:# and reward_e == 0:
            reward_e += self.time_step/50
        
        self.obs = np.concatenate([next_obs_p1,next_obs_p2,next_obs_e],0) 

        reward = np.array([reward_p1, reward_p2, reward_e]) 

        if self.time_step >= self.episode_limit:
            infos = {'touch_reward': reward, "episode_limit": True}
        else:
            infos = {'touch_reward': reward}
        # if self.time_step == 2:
        #import pdb; pdb.set_trace()
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

    def get_reward_pursuer(self, abs_pos_own, abs_pos_mate, abs_pos_adv, reward_share):
        dist1 = get_dist(abs_pos_own, abs_pos_adv)
        dist2 = get_dist(abs_pos_mate, abs_pos_adv)
        reward = 0

        if reward_share == True:
            if dist1 < 0.15 or dist2 < 0.15:
                reward = 1
            elif np.abs(abs_pos_own[0]) > 1.1 or np.abs(abs_pos_own[1]) > 1.1:
                reward = self.reward_out
        elif reward_share == False:
            try:
                if dist1 < 0.15:
                    reward = 1
                elif np.abs(abs_pos_own[0]) > 1.1 or np.abs(abs_pos_own[1]) > 1.1:
                    reward = self.reward_out
            except:
                import pdb; pdb.set_trace()

        return reward


    def get_reward_evader(self, abs_pos_own, abs_pos_adv1, abs_pos_adv2):
        dist1 = get_dist(abs_pos_own, abs_pos_adv1)
        dist2 = get_dist(abs_pos_own, abs_pos_adv2)
        reward = 0

        #if dist1 < 0.15 or dist2 < 0.15:
        #    reward = -1
        #el
        if np.abs(abs_pos_own[0]) > 1.1 or np.abs(abs_pos_own[1]) > 1.1:
            reward = self.reward_out

        return reward

    def get_done(self, abs_pos_own, abs_pos_adv1, abs_pos_adv2, num_step, max_step):
        dist1 = get_dist(abs_pos_own, abs_pos_adv1)
        dist2 = get_dist(abs_pos_own, abs_pos_adv2)

        if dist1 < 0.15 or dist2 < 0.15 or num_step > max_step or  \
            np.abs(abs_pos_own[0]) > 1.1 or np.abs(abs_pos_own[1]) > 1.1 or \
            np.abs(abs_pos_adv1[0]) > 1.1 or np.abs(abs_pos_adv1[1]) > 1.1 or \
            np.abs(abs_pos_adv2[0]) > 1.1 or np.abs(abs_pos_adv2[1]) > 1.1:
            done = True
        else:
            done = False

        return done
