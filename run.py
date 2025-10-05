import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, ReplayBuffer_Prior
from components.transforms import OneHot

import numpy as np
import copy as cp
import pandas as pd
import random
from functools import partial
from components.episode_buffer import EpisodeBatch

os.environ["OMP_NUM_THREADS"]=str(4) 

def run(_run, _config, _log):
    
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = args.GPU if args.use_cuda else "cpu"
    
    # args.buffer_size = 32 # temporally
    args.lambda1 = 1. # TD 
    args.lambda2 = 1e-5 # L2
    if args.alg_AAS: # 
        if 'animarl_agent' in args.env_args['env_name'] or 'animarl_silkmoth' in args.env_args['env_name']:
            args.lambda3 = 10 # domain_adapt
        elif 'animarl_fly' in args.env_args['env_name']:
            args.lambda3 = 5 
        elif 'animarl_newt' in args.env_args['env_name']:
            args.lambda3 = 1 
    elif args.alg_DIL: # 
        if 'animarl_agent' in args.env_args['env_name'] or 'animarl_silkmoth' in args.env_args['env_name']:
            args.lambda3 = 1 # domain_adapt
        elif 'animarl_fly' in args.env_args['env_name']:
            args.lambda3 = 1/2 
        elif 'animarl_newt' in args.env_args['env_name']:
            args.lambda3 = 1/10 
    else: 
        args.lambda3 = 1

    # condition predition for CF
    if 'animarl_agent' in args.env_args['env_name']:
        args.lambda4 = 0.01
    else:
        args.lambda4 = 0.01 

    if 'animarl' in args.env_args['env_name']:
        args.multi_env = True
    else: 
        args.multi_env = False
    # setup loggers
    logger = Logger(_log)
    
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    if args.cont != 'cont' and args.cont is not False:
        unique_token = "{}__{}".format(
            args.name, args.cont)
    else:
        unique_token = "{}/{}/seed_{}".format(
            args.env_args['env_name'], args.name, args.seed)
        #unique_token = "{}__{}".format(
        #    args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs")
        env_name = args.env_args['env_name']

        tensorboard_dir = f'{tb_logs_direc}/{args.name[:-6]}/{env_name}/seed_{args.seed}'
        logger.setup_tb(tensorboard_dir)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential(args, logger):

    args.gw = True if "GW" in args.name else False 
    args.dtw = True if "DTW" in args.name else False 
    args.dueling = False if args.alg_BC else True

    args.action_shape = 1

    args.result_path = '../AniMARL_results'
    load_path2 = '../AniMARL_results/preprocessed/'   

    if "animarl_agent_2vs1" in args.env:
        args.burn_in = 0
        filename = 'agent'
        args.epsilon_finish = 0.1 
        args.epsilon_test = 0.1 
        args.epsilon_start_finetune = 0.3 
        args.epsilon_finish_finetune = 0.1
        args.lr_adam = 1e-6 
        args.lr_adam_finetune = 1e-7 
    elif "silkmoth" in args.env:
        args.burn_in = 10 
        filename = 'silkmoth_left' if args.variant == '_Left' else 'silkmoth'
        images_ = []
        for j in range(5):
            filename_video = load_path2+ "silkmoth_video_"+str(j)+"_2.0_Hz.npz" # 
            images_.append(np.load(filename_video, allow_pickle=True)["arr_0"])
        args.epsilon_finish = 0.3 
        args.epsilon_start_finetune = 0.5 
        args.epsilon_finish_finetune = 0.3 
        args.epsilon_test = 0.5 
        args.lr_adam = 1e-7 
        args.lr_adam_finetune = 1e-8 
    elif "fly" in args.env:
        args.burn_in = 0
        filename = 'fly'
        args.epsilon_finish = 0.3 
        args.epsilon_start_finetune = 0.5 
        args.epsilon_finish_finetune = 0.3 
        args.epsilon_test = 0.5  
        args.lr_adam = 0.00001  
        args.lr_adam_finetune = 0.000001 
    elif "newt" in args.env:
        args.burn_in = 10
        filename = 'newt'
        args.epsilon_finish = 0.3 
        args.epsilon_start_finetune = 0.5 
        args.epsilon_finish_finetune = 0.3 
        args.epsilon_test = 0.5 
        args.lr_adam = 0.00001 
        args.lr_adam_finetune = 0.000001  
    
    if ('CF' in args.cond) and "animarl_agent_2vs1" in args.env:
        preprocessed = np.load(load_path2+filename+'_400_CF.npz', allow_pickle=True)
    elif ('CF' in args.cond) and "silkmoth" in args.env:
        preprocessed = np.load(load_path2+filename+'_48_CF.npz', allow_pickle=True)    
    else:   
        filename += '_'+args.cond  # .replace('_CF','')
        preprocessed = np.load(load_path2+filename+'.npz', allow_pickle=True)

    states = preprocessed['arr_0']
    actions = preprocessed['arr_1']
    rewards = preprocessed['arr_2']
    lengths = preprocessed['arr_3']
    condition = preprocessed['arr_4']
    us = preprocessed['arr_5']
    ds = preprocessed['arr_6']
    states_ = states.copy()
    actions_ = actions.copy()
    rewards_ = rewards.copy()
    lengths_ = lengths.copy()
    condition_ = condition.copy()

    if "animarl_agent_2vs1" in args.env or "fly" in args.env or "newt" in args.env:
        args.params = {
            'mass_p1': 1,
            'speed_p1': us[0],
            'damping_p1': ds[0],
            'mass_p2': 1,
            'speed_p2': us[1],
            'damping_p2': ds[1],
            'mass_e': 1,
            'speed_e': us[2],
            'damping_e': ds[2],
        }
    elif "silkmoth" in args.env:
        args.params = {
            'mass': 1,
            'speed': us[0],
            'damping': ds[0],
            'origin': [0,0],
        }
    elif "bat" in args.env:
        import pdb; pdb.set_trace()         
    elif "dragonfly" in args.env:
        import pdb; pdb.set_trace()        

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_all_agents = env_info["n_agents"]
    # args.n_actions = env_info["n_actions"]
    args.n_actions = runner.env.n_actions
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]


    if "animarl" in args.env:
        args.reward_dim = runner.env.reward_dim
        args.hidden_dim = 64
        if args.alg_RNN:
            args.agent = 'rnn'
            args.hidden_depth = 1
        else:
            args.agent = 'mlp' # 


    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (args.action_shape,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": th.int}, # env_info["n_actions"]
        "reward": {"vshape": (args.reward_dim,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }    

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              args.burn_in_period, args, from_demo=args.from_demo,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)


    args.n_all_agents = runner.env.n_all_agents
    

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    
    
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    on_policy_episode = 0

    start_time = time.time()
    last_time = start_time


    # save_path 
    save_path = os.path.join(args.local_results_path, "models", args.unique_token) # , str(runner.t_env)

    #if args.cont == 'data':
    #    save_path += '_pretrained'
    # if args.cont == 'cont' or args.cont is False:
    if args.alg_RNN:
        save_path += '_RNN'
    if args.pretrain:
        save_path += '_pretrain'
    if args.from_demo:
        save_path += '_from_demo'
    if args.alg_AAS:
        save_path += '_AAS'
    if args.alg_DIL:
        save_path += '_DIL'
    if args.alg_BC:
        save_path += '_BC'

    # if 'CF2' in args.cond:
    #    save_path += '-ts-CF'
    if 'CF3' in args.cond or 'CF4' in args.cond or 'CF5' in args.cond or 'CF6' in args.cond:
        if 'CF3' in args.cond or 'CF5' in args.cond:
            if "silkmoth" in args.env:
                save_path += '-ts-48'
            elif "agent" in args.env:
                save_path += '-ts-400'
        elif 'CF4' in args.cond or 'CF6' in args.cond:
            save_path += '-ts-CF'

    else:
        save_path += '-ts-'+args.cond   
    save_path += args.variant

    if args.cont == 'cont' or args.test: 
        os.makedirs(save_path, exist_ok=True)
        savefolders = os.listdir(save_path+'/')
        if False: # args.double_q:
            save_path += '/'+[s for s in savefolders if '_x' in s][-1]
        else: 
            try: save_path += '/'+[s for s in savefolders if not ('_x' in s)][-1]
            except: import pdb; pdb.set_trace()

        logger.console_logger.info("load path: "+save_path)
    else:
        print("save path: "+save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        

    if "animarl" in args.env:
        args.initial = True

        if "silkmoth" in args.env:
            n_train = 36 
            n_test = 23 
            n_test_runs = 115
            args.n_dfs = 0 
            filename = 'animarl_silkmoth'
            args.action_dim = 13
        elif args.env == "animarl_agent_2vs1":
            n_train = 50 
            n_test = 50 
            n_test_runs = 250
            args.n_dfs = 1
            # state_dim = 12
            filename = 'agent'
            args.action_dim = 13
        elif "fly" in args.env:
            n_train = 50 
            n_test = 20
            n_test_runs = 100
            args.n_dfs = 1 
            filename = 'animarl_fly'
            args.action_dim = 13
        elif "newt" in args.env:
            n_train = 50 
            n_test = 20
            n_test_runs = 100
            args.n_dfs = 1 
            filename = 'animarl_newt'
            args.action_dim = 13
  

        args.test_nepisode = n_test_runs
        # initials = np.zeros((len(runner.n_demo),state_dim))

        # if args.env == "animarl_agent_2vs1": #  or args.env == "animarl_fly_2vs1" or args.env == "animarl_newt_2vs1": 
        #   runner.n_demo = np.random.permutation(runner.n_demo) # randomize
        if args.env == "animarl_silkmoth":
            images = []

        states,actions,rewards,lengths,conditions,initials,avail_actions = [],[],[],[],[],[],[]
        # ii = 0
        
        for i in runner.n_demo: 
            if args.cond == 'CF3' or args.cond == 'CF4' or 'CF5' in args.cond or 'CF6' in args.cond:
                if states_[i][0,-1]==0:
                    states_[i][:,-1] = 1
                elif states_[i][0,-1]==1:
                    states_[i][:,-1] = 0

            states.append(states_[i])
            action_onehot = np.identity(args.action_dim)[actions_[i]]
            actions.append(action_onehot)
            rewards.append(rewards_[i])
            lengths.append(lengths_[i])
            conditions.append(condition_[i])
            initials.append(states_[i][0:1])
            avail_actions.append(np.ones_like(action_onehot))

            if args.env == "animarl_silkmoth":
                q_ = int(np.floor(i/6))
                if q_ <= 1:
                    q__ = 0
                elif q_ <= 3:
                    q__ = 1
                elif q_ <= 5:
                    q__ = 2
                elif q_ == 6 or q_ == 8:
                    q__ = 3
                elif q_ == 7 or q_ == 9:
                    q__ = 4
                images.append(images_[q__])
            # seqs.append(i)
            # ii += 1
        print(conditions)          
        print(lengths)
        
        # if np.min(np.array(lengths)) <20:
        #    import pdb; pdb.set_trace()

        # if args.from_demo:
        new_batch = partial(EpisodeBatch, scheme, groups, 1, args.episode_limit + 1, # args.batch_size
                            preprocess=preprocess, device=args.device)
        from utils.utils_chase import get_observation_from_state
    else: 
        args.initial = False
    
    print(args)
    print('initial value from data: '+str(args.initial))
    # n_test_runs = n_test
    # train 
    if not args.test:
        if args.cont == 'cont':
            learner.load_models(save_path)
            runner.t_env = int(save_path.split('/')[-1])
            logger.console_logger.info(
                "Restart training from {} time steps for total {} timesteps".format(runner.t_env,args.t_max))
        elif args.cont == False:
            logger.console_logger.info(
                "Beginning training for {} timesteps".format(args.t_max))
        if args.pretrain: 
            
            if args.alg_BC:
                model_path = os.path.join(args.result_path,'model_BC')
            else:
                model_path = os.path.join(args.result_path,'model_Q/')

            mixer = None
            opt = False # True if args.cont == 'all' else False
            if args.alg_RNN:
                str_rnn = 'RNN_'
            else:
                str_rnn = ''

            if args.env == "animarl_agent_2vs1" or args.env == "animarl_fly_2vs1" or args.env == "animarl_newt_2vs1" :
                reward_str = 'touch'
                
            elif args.env == "animarl_silkmoth":
                reward_str = 'reach'
                
 
            if args.alg_AAS:
                if args.env == "animarl_agent_2vs1" or args.env == "animarl_silkmoth" :
                    param_str = 'lmd2-50.0-lr-0.001'
                elif args.env == "animarl_fly_2vs1": 
                    param_str = 'lmd2-10.0-lr-0.0001'
                elif args.env == "animarl_newt_2vs1" :
                    param_str = 'lmd2-10.0-lr-0.0001'
                load_path = os.path.join(model_path,'DQN_'+str_rnn+reward_str+'_AS-'+args.env+'-ts-'+args.cond+'-'+param_str)
            elif args.alg_DIL:
                if 'CF' in args.cond:
                    param_str = 'lmd2-0.01-lr-0.0001'
                    if args.env == "animarl_agent_2vs1":
                        cond_ = '400'
                    elif args.env == "animarl_silkmoth" :
                        cond_ = '48'
                    
                    load_path = os.path.join(model_path,'DQN_'+str_rnn+reward_str+'_DIL_CF-'+args.env+'-ts-'+cond_+'-'+param_str)
                else:
                    if args.env == "animarl_agent_2vs1" or args.env == "animarl_silkmoth" :
                        param_str = 'lmd2-10.0-lr-0.0001'
                    elif args.env == "animarl_fly_2vs1": 
                        param_str = 'lmd2-1.0-lr-0.0001'
                    elif args.env == "animarl_newt_2vs1" :
                        param_str = 'lmd2-0.5-lr-0.0001'
                    load_path = os.path.join(model_path,'DQN_'+str_rnn+reward_str+'_DIL-'+args.env+'-ts-'+args.cond+'-'+param_str)
            elif args.alg_BC:
                load_path = os.path.join(model_path,'RNN_'+reward_str+'-'+args.env+'-ts-'+args.cond+'-lr-0.001')
            #else:
            #    import pdb; pdb.set_trace()

            learner.load_models(load_path,pretrain=args.pretrain,mixer=mixer,opt=opt)
            logger.console_logger.info(
                "Load pretrained model from {} and train for total {} timesteps".format(load_path,args.t_max))        

        actions_demo = []
        nn = 0 
        t_env_0 = 0 if not (args.cont is not True) else runner.t_env
        while runner.t_env <= args.t_max:
            # Run for a whole episode at a time
            n = np.mod(nn,n_train)# runner.n_demo[]
            if args.initial:
                initial = initials[n]
                state_demo = states[n]
                if args.from_demo:
                    demo_batch = new_batch()
                    selfpos = False if args.env == "animarl_silkmoth" else True
                    selfdist = True if "animarl_fly" in args.env else False
                    obs = get_observation_from_state(states[n],args.n_agents,runner.env.n_all_agents-args.n_dfs,args.n_dfs,selfpos=selfpos,selfdist=selfdist)
                    if args.env == "animarl_agent_2vs1": # or args.env == "animarl_silkmoth":
                        obs = np.concatenate([obs,np.repeat(states[n][None,:,-1:],args.n_agents,axis=0)],axis=2)
                    # actions_demo = actions[n]
                    action_demo = actions[n][:,:args.n_agents]
                    actions_demo.append(np.pad(action_demo,[(0,args.episode_limit-state_demo.shape[0]),(0,0),(0,0)], 'constant')) #
                    for t in range(lengths[n]):
                        states_ = states[n][None,t] if "animarl" in args.env else states[n][t]
                        pre_transition_data = {
                                "state": [states_], # [1,feat]
                                "avail_actions": [avail_actions[n][t,:args.n_agents]], # [agents][actions]
                                "obs": [obs[None,:,t]], # [agents,feat]
                            }
                        demo_batch.update(pre_transition_data, ts=t)
                        reward_ = rewards[n]

                        actions_ = np.where(action_demo[t]==1)[1]
                        if args.n_agents == 1:
                            actions_ = actions_[:,None] 

                        post_transition_data = {
                            "actions": actions_, # [1,agents]
                            "reward": [(reward_,)], # int 
                            "terminated": [(True if t == lengths[n]-1 else False,)], # bool
                        }
                        demo_batch.update(post_transition_data, ts=t)

                    buffer.insert_episode_batch(demo_batch, demo=True)

                else:
                    actions_demo = None
                    action_demo = actions[n][:,:args.n_agents]

            else:
                initial,actions_demo = None,None
                n = nn # runner.t_env
                action_demo = actions[n][:,:args.n_agents]

            if args.env == "animarl_silkmoth":
                additional_info = images[n]
            else:
                additional_info = None

            episode_batch = runner.run(n=n, states_demo=state_demo,test_mode=False,action_demo=action_demo,info=additional_info)
            buffer.insert_episode_batch(episode_batch)

            for ii in range(args.num_circle):
                if buffer.can_sample(args.batch_size): # False first

                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    
                    actions_demo = np.array(actions_demo)
                    learner.train(episode_sample, runner.t_env, episode, use_demo=args.from_demo, actions_demo=actions_demo)
                    actions_demo = []

            nn += 1                
            # Execute test runs once in a while
            # n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                
                logger.console_logger.info(
                    "t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()

                last_test_T = runner.t_env
                for n_te in range(n_test_runs):
                    if args.initial:
                        n_te = n_train+np.mod(n_te,n_test)# runner.n_demo[n_train:][] -n_test
                        state_demo = states[n_te]
                    else:
                        state_demo = None
                    if args.env == "animarl_silkmoth":
                        additional_info = images[n_te]
                    else:
                        additional_info = None
                    if args.env == "animarl_silkmoth" or args.env == "animarl_newt_2vs1":
                        action_demo = actions[n_te]
                    else:
                        action_demo = None
                    runner.run(n=n_te,states_demo=state_demo, test_mode=True, action_demo=action_demo, info=additional_info)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                #save_path = os.path.join(
                #    args.local_results_path, "models", args.unique_token, str(runner.t_env))
                # "results/models/{}".format(unique_token)
                #if args.pretrain:
                #    save_path += '_pretrain'
                #if args.from_demo:
                #    save_path += '_from_demo'
                #if args.cont != "cont" and args.cont != False:
                #    save_path += '_pretrain_' + args.cont
                save_path2 = save_path + '/' + str(runner.t_env)
                os.makedirs(save_path2, exist_ok=True)
                #if args.double_q:
                #    os.makedirs(save_path + '_x', exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path2)) 

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path2)

            episode += args.batch_size_run * args.num_circle

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        if not "animarl" in args.env:
            runner.close_env()
        logger.console_logger.info("Finished Training")
        learner.load_models(save_path2)

    else: # test
        learner.load_models(save_path)

    # Execute test runs once in a while
    import glob
    runner.batch_size = 1
    # n_test_runs = max(1, args.test_nepisode // runner.batch_size)

    statess,avail_actionss,obss,actionss,rewardss,times,agent_outputs,array_csvs = [],[],[],[],[],[],[],[]
    states_demo, actions_demo, lengths_demo, rewards_demo = [],[],[],[] 

    save = True
    if save: 
        import matplotlib.pyplot as plt
        from matplotlib import animation
        import matplotlib
        matplotlib.use('Agg')
        import pickle

    filename = './results/results/'+args.env+'/'+args.name+'/'+str(args.seed)#+'/'
    if args.alg_RNN:
        filename += '_RNN'
    if args.pretrain:
        filename += '_pretrain'
    if args.from_demo:
        filename += '_from_demo'
    if args.alg_AAS:
        filename += '_AAS'
    if args.alg_DIL:
        filename += '_DIL'
    if args.alg_BC:
        filename += '_BC'
    filename += '-ts-'+args.cond
    filename += args.variant
    

    os.makedirs(filename, exist_ok=True)

    if not "animarl" in args.env:
        interval = 200 # default
    else:
        interval = 200

    for n in range(n_test_runs):
        print(str(n)+'/'+str(n_test_runs))
        if args.initial:
            n_ = n_train+np.mod(n,n_test)# runner.n_demo[n_train:][]-n_test
            # initial = initials[n_train+n_]
            try: print(runner.n_demo[n_])
            except: import pdb; pdb.set_trace()
            state_demo = states[n_]
            action_demo_ = actions[n_]
            action_demo = np.where(actions[n_][:,:args.n_agents]==1)[2].reshape((-1,args.n_agents))
            length_demo = lengths[n_]
            reward_demo = rewards[n_]
        else:
            state_demo = None
            action_demo_ = None
            n_ = n

        if args.env == "animarl_silkmoth":
            additional_info = images[n_]
        else:
            additional_info = None

        if save: 
            state,avail_action,obs,action,reward,t,agent_output,dist = runner.run(
                n=n_, states_demo=state_demo,test_mode=True, action_demo=action_demo_, info=additional_info, save=save)
        else:
            runner.run(n, states_demo=state_demo, test_mode=True, info=additional_info, save=save)
            list_of_files = glob.glob('./dumps/'+str(args.seed)+'/*') 
            latest_file = max(list_of_files, key=os.path.getctime)
            print('use latest file: '+latest_file)
            script_helpers.ScriptHelpers().dump_to_video(latest_file) # replay()
        
        # append
        if save: # and np.max(reward)>0:
            statess.append(state)
            avail_actionss.append(avail_action)
            obss.append(obs)
            actionss.append(action)
            rewardss.append(reward)
            times.append(t)
            agent_outputs.append(agent_output)

            states_demo.append(state_demo) 
            actions_demo.append(action_demo) 
            lengths_demo.append(length_demo) 
            rewards_demo.append(reward_demo)

            if 'animarl' in args.env: 
                state = state[:,0]
            if len(action.shape) == 3:
                action = action[:,0]

            # RL 
            fig = plt.figure(figsize=(8,6)) # 8,6
            fig.subplots_adjust(bottom = 0.2)
            ax = fig.add_subplot(211) # 211

            filename2 = filename + '/run_'+str(n)+'_score_'+str(np.round(np.max(reward),1))
            #if np.max(reward)>0:
            #    import pdb; pdb.set_trace()
            # print('create video: '+filename)
            demo_flag = False

            # output csv
            array_csv = np.concatenate([reward[-1][0],reward_demo])
            reward_pred = 1 if np.all(reward[-1][0] == reward_demo) else 0
            
            if args.n_all_agents == 1:
                dist1 = np.sum(np.linalg.norm(state[:, 2:4], axis=1))/2
                dist1_GT = np.sum(np.linalg.norm(state_demo[:, 2:4], axis=1))/2
                array_csv = np.concatenate([array_csv,np.array([reward_pred]),np.array([dist]),np.array([state.shape[0]/2,state_demo.shape[0]/2,dist1,dist1_GT])])
            else:
                # if args.env == "animarl_agent_2vs1":
                dist1 = np.sum(np.linalg.norm(state[:, 2:4], axis=1))/10
                dist2 = np.sum(np.linalg.norm(state[:, 6:8], axis=1))/10
                dist3 = np.sum(np.linalg.norm(state[:, 10:12], axis=1))/10
                dist1_GT = np.sum(np.linalg.norm(state_demo[:, 2:4], axis=1))/10
                dist2_GT = np.sum(np.linalg.norm(state_demo[:, 6:8], axis=1))/10
                dist3_GT = np.sum(np.linalg.norm(state_demo[:, 10:12], axis=1))/10
                
                array_csv = np.concatenate([array_csv,np.array([reward_pred]),np.array([dist,dist1,dist2,dist3,dist1_GT,dist2_GT,dist3_GT,state.shape[0]/10,state_demo.shape[0]/10])])
                #else:
                #    array_csv = np.concatenate([array_csv,np.array([reward_pred]),np.array([dist])])
            array_csvs.append(array_csv)

            # output video
            vid_fig = False
            if vid_fig:
                ani = animation.FuncAnimation(fig, update_func, fargs = (ax,state,action,agent_output,args.env,demo_flag,fig), interval=interval, frames=state.shape[0], repeat=False) # interval=100,
                ani.save(filename2+'.mp4',writer="ffmpeg", fps=10) # ani.save(filename2+'.gif',writer="pillow", fps=10)
                # filename += ".mp4"; ani.save(filename, writer="ffmpeg")
            else:
                update_contents(state.shape[0]-1, ax,state,action,agent_output,args.env,demo_flag,fig)
                fig.savefig(filename2+'.png')
            
            plt.clf(); plt.close()

            if True: # action_demo is not None: # demo
                fig = plt.figure(figsize=(8,3))
                fig.subplots_adjust(bottom = 0.2)
                ax = fig.add_subplot(111)
                ax.set_aspect('equal') 
                demo_flag = True
                if vid_fig:
                    ani = animation.FuncAnimation(fig, update_func, fargs = (ax,state_demo,action_demo,None,args.env,demo_flag,fig), interval=interval, frames=length_demo, repeat=False) # interval=100,       
                    ani.save(filename2+'_demo.mp4',writer="ffmpeg", fps=10)  #ani.save(filename2+'_demo.gif',writer="pillow", fps=10)
                    # filename += ".mp4"; ani.save(filename, writer="ffmpeg")
                else:
                    update_contents(state_demo.shape[0]-1, ax,state_demo,action_demo,None,args.env,demo_flag,fig)
                    fig.savefig(filename2+'_demo.png')
                plt.clf(); plt.close()
            

    if save:
        result = [statess,avail_actionss,obss,actionss,rewards,times,agent_outputs,states_demo, actions_demo, lengths_demo, rewards_demo]
        # resultfile = './results/results/'+str(args.seed) # +'/result'
        with open(filename +'/result', mode='wb') as f:
            pickle.dump(result, f, protocol=4)
        # save csv
        array_csvs = np.array(array_csvs)
        filename_csv = filename + '/summary.csv'
        if args.n_all_agents == 3:
            # if args.env == "animarl_agent_2vs1":
            header = ['r_0_sim','r_1_sim','r_2_sim','r_0_demo','r_1_demo','r_2_demo','reward_corr','dist','dist_1','dist_2','dist_3','dist_1_GT','dist_2_GT','dist_3_GT','time','time_GT']
            #else:
            #    header = ['r_0_sim','r_1_sim','r_2_sim','r_0_demo','r_1_demo','r_2_demo','reward_corr','dist']
        elif args.n_all_agents == 1:
            header = ['r_0_sim','r_0_demo','reward_corr','dist','time','time_GT','dist_1','dist_1_GT']
        else:
            import pdb; pdb.set_trace()
        df = pd.DataFrame(array_csvs)
        df.to_csv(filename_csv, header=header, index=True)

    logger.console_logger.info("Finished Test")
    import pdb; pdb.set_trace()
    
def update_func(i, ax, state, action, agent_output, env, demo, fig):

    update_contents(i, ax, state, action, agent_output, env, demo, fig)

def update_contents(i, ax, state, action, agent_output, env, demo, fig):
    win = 99999
    # win = 3 if i < 10 else 10

    if 'silkmoth' in env: 
        n_mate = 1; n_opponent = 0 
    elif 'animarl_agent' in env or 'animarl_fly' in env or 'animarl_newt' in env: 
        if state.shape[1] == 12:
            n_mate = 2; n_opponent = 1 

    # ax = fig.add_subplot(1,n_mate+2,1)
    ax.set_aspect('equal')
    ax.clear()

    if 'animarl_agent' in env or 'animarl_newt' in env: 
        top = [-1,-1]
        bottom = [1,1]
        top_bottom = [-1,1]
        left = [-1, -1]
        right = [1, 1]
        left_right = [-1, 1]
        ax.plot(left, top_bottom, color="black")
        ax.plot(right, top_bottom, color="black")
        ax.plot(left_right, top, color="black")
        ax.plot(left_right, bottom, color="black")
    elif 'animarl_silkmoth' in env: 
        top = [0.3,0.3]
        bottom = [-0.3,-0.3]
        top_bottom = [-0.3,0.3]
        left = [-0.1, -0.1]
        right = [0.4, 0.4]
        left_right = [-1.25, 1.25]
        ax.plot(0, 0, marker='x',color="black")
    elif  'animarl_fly' in env:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, color="black")

    # n_agents = int((state.shape[1]-6)/4)
    if 'animarl_silkmoth' in env: 
        n = 0
        ax.plot(state[i][n*6], state[i][n*6+1], 'o', markersize=8, color="gray")
        ax.plot(state[i-win:i+1,n*6], state[i-win:i+1,n*6+1], '.', markersize=4, color="gray", alpha=0.3)
        ax.text(state[i][n*6], state[i][n*6+1], str(n)) # size=30)

    elif 'animarl_agent' in env or 'animarl_fly' in env or 'animarl_newt' in env: 
        if state.shape[1] == 12 or state.shape[1] == 13:
            n_mate = 2; n_opponent = 1 
    
        for n in range(n_mate):
            ax.plot(state[i][n*4], state[i][n*4+1], 'o', markersize=8, color="gray")
            ax.plot(state[i-win:i+1,n*4], state[i-win:i+1,n*4+1], '.', markersize=4, color="gray", alpha=0.3)
            ax.text(state[i][n*4], state[i][n*4+1], str(n)) # size=30)
        for n in range(n_opponent):
            ax.plot(state[i][n_mate*4+n*4], state[i][n_mate*4+n*4+1], 'o', markersize=8, color="black")
            ax.plot(state[i-win:i+1,n_mate*4+n*4], state[i-win:i+1,n_mate*4+n*4+1], '.', markersize=4, color="black", alpha=0.3)
    
    action_str = '' 

    if action is not None:
    
        action_ = action[i]
    
        if action.shape[-1]==3 and action.shape[-2]==n_mate+n_opponent:
            action_str += str(action_[n]) 
        else:
            for n in range(n_mate): 
                action_str += str(n)+': '+str(action_[n])+', '

    ax.set_title('frame: '+str(i)+' (10 Hz), action '+action_str)

    
    if 'animarl_agent' in env or 'animarl_fly' in env or 'animarl_newt' in env: 
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    elif 'animarl_silkmoth' in env: 
        ax.set_xlim(-0.1, 0.4)
        ax.set_ylim(-0.3, 0.3)
    #ax.set_xlabel("x-position")
    #ax.set_ylabel("y-position")
    ax.set_aspect('equal')
    

    # q-value
    if False: # agent_output is not None:
        agent_output = np.array(agent_output)
        n_actions = agent_output.shape[-1]
        red = [177/255, 24/255, 42/255]
        darkblue = [4/255, 44/255, 88/255]
        blue = [31/255, 100/255, 169/255]
        lightblue = [65/255, 144/255, 194/255]
        qmin, qmax = np.min(agent_output), np.max(agent_output)

        xs = range(n_actions)
        for n in range(n_mate): 
            output = agent_output[i,0,n]
            # ax = fig.add_subplot(1,n_mate+2,n+2)
            
            ax = fig.add_subplot(2,n_mate,n_mate+n+1)    
            ax.set_title('agent '+str(n))
            try: ax.bar(xs,  output, color=darkblue, alpha=0.7)
            except: import pdb; pdb.set_trace()
            [ax.spines[side].set_visible(False) for side in ['right', 'top']]
            ax.set_ylim(qmin, qmax)
            if n_actions == 13:
                ax.set_xticks( np.arange(0, 13, 2))
            elif n_actions == 12:
                ax.set_xticks( np.arange(0, 12, 2))
            else:
                import pdb; pdb.set_trace()

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.infoing(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
