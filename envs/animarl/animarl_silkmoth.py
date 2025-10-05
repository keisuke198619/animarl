from .. import MultiAgentEnv
import gym
import numpy as np
from utils.utils_chase import *

class Animarl_Silkmoth(MultiAgentEnv):

    def __init__(
        self,
        n_agents=1,
        time_limit=3000,
        time_step=0,
        obs_dim=10,
        env_name='animarl_silkmoth',
        stacked=False,
        representation="simple",
        rewards='touch',
        logdir='animarl_silkmoth_dumps',
        args=None,
        number_of_opponents_controls=0,
        number_of_opponents=0,
        seed=0,
        action_dim=13,
        state_dim=12,
        space_dim=2,
        reward_dim=1,
        dt = 0.5,
        reward_share=False # [0.001,0.4]
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
        
        self.n_all_agents = 1
        self.n_mate = n_agents-number_of_opponents
        self.n_adv = number_of_opponents

        self.speed = args.params["speed"] 
        self.mass = args.params["mass"]
        self.damping = args.params["damping"]
        self.origin = args.params["origin"]
        self.dt = dt
        self.reward_share = reward_share
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.space_dim = space_dim
        self.tdl = 0
        self.tdr = 0
        self.rstimL = np.zeros((3,))
        self.rstimR = np.zeros((3,))
        self.alphal = 0
        self.alphar = 0
        self.action_space = [gym.spaces.Discrete(
            action_dim) for _ in range(self.n_agents)]

        self.unit_dim = self.obs_dim # QPLEX unit_dim for cds_gfootball

        self.n_actions = self.action_space[0].n    

    def get_simple_obs(self, index=-1):

        if index == -1:
            simple_obs = self.state[None]         

        else:
            simple_obs = self.obs[index]
        
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)
    
    def step(self, actions, images):
        """Returns reward, terminated, info."""
        self.time_step += 1

        action_p1 = actions.detach().to('cpu').numpy().tolist()[0]

        params = {
            'mass': self.mass,
            'speed': self.speed,
            'damping': self.damping,
            'dt': self.dt,
            'n_mate': self.n_mate,
            'n_adv': self.n_adv,
            'origin': np.array(self.origin)
        }

        next_state, next_pos = transition_single(self.state, action_p1, params)

        # odor_wind_vision = self.mothsens_extractor(self.state, next_state, images[self.time_step], self.origin)
        src_angle = angle_vel_pos_silkmoth(self.state[:2], self.state[2:4])
        next_src_angle = angle_vel_pos_silkmoth(next_state[0], next_state[1])
        tdlr_rstimLR = [self.tdl, self.tdr, self.rstimL, self.rstimR, self.alphal, self.alphar]

        odor_vis_wind,tdlr_rstimLR = compute_odor_vis_wind_silkmoth(src_angle[0], next_src_angle[0],
                                next_pos, self.state[5:7], images[self.time_step], tdlr_rstimLR, self.time_step, self.dt, str(self.state[-1]))

        self.tdl, self.tdr, self.rstimL_prev, self.rstimR_prev, self.alphal, self.alphar = tdlr_rstimLR

        state_p1 = np.concatenate([next_state.reshape((4,1)), src_angle[:,None], odor_vis_wind[:,None],self.cond[:,None]]) # [posx2, velx2, anglex1, odorx2, visionx1, windx4]
        
        reward_p1 = self.get_reward_pursuer(next_pos) 
 
        done = self.get_done(next_pos, self.time_step, self.episode_limit)
        
        self.state = state_p1[:,0]
        
        self.obs = [self.state[2:]] # remove position

        reward = np.array([reward_p1])
 
        # if self.time_step >= self.episode_limit:
        infos = {'touch_reward': reward, "episode_limit": True}
        # else:
        #     infos = {'touch_reward': reward+self.time_step/600}
        # if self.time_step == 2:
        #   import pdb; pdb.set_trace()

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

        # state_p1 = initial # get_obs_p(state_p1,self.n_mate,self.n_adv) # obs_p1

        self.state = initial  
        self.obs = [initial[2:]] # remove position
        self.cond = initial[-1:]


        return self.obs, self.state 

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_abs_u(self, action, abs_own_pos, abs_adv_pos, para_speed, sign=1):
        if sign == 1:
            th_stop = 1
        elif sign == -1:
            th_stop = 1

        if np.abs(action[0]) < th_stop:
            action0 = action[0]/th_stop # rescaling 
            mean_speed = (para_speed[0]+para_speed[1])/2
            range_speed = np.abs(para_speed[1]) - mean_speed
            speed = mean_speed + action0*range_speed

            if sign == 1:
                ang = action[1] * np.pi/2 # [-pi/2 pi/2] (previous: action * -np.pi / 6)
            elif sign == -1: # need to modify when the opposite direction
                ang = action[1] * np.pi/2 + np.pi # [pi 3*pi/2]

            sub_u = [np.cos(ang), np.sin(ang)]            
            abs_u = speed*rotate_u(sub_u, abs_own_pos, abs_adv_pos)
        else:
            abs_u = [0, 0]
    
        return abs_u

    def get_next_own_state(self, abs_pos_own, abs_vel_own, abs_u, mass, var_damping, para_damping, dt):
        abs_acc_own = np.array(abs_u) / mass
        mean_damping = (para_damping[0]+para_damping[1])/2
        range_damping = para_damping[1] - mean_damping
        damping = mean_damping + var_damping*range_damping

        next_abs_vel_own = abs_vel_own * (1 - damping) + abs_acc_own * dt
        next_abs_pos_own = abs_pos_own + next_abs_vel_own * dt
        return next_abs_pos_own, next_abs_vel_own


    def get_reward_pursuer(self, abs_pos_own):
        dist1 = np.sqrt(np.sum(abs_pos_own**2))
        reward = 0

        if dist1 < 0.05:
            reward = 1

        return reward


    def get_done(self, abs_pos_own, num_step, max_step):
        dist1 = np.sqrt(np.sum(abs_pos_own**2))

        if dist1 < 0.05 or num_step > max_step:
            done = True
        else:
            done = False

        return done
