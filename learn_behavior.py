import os, sys, glob, argparse, random, copy, math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import multiprocessing as mp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pretrain.data_loader_agent_2vs1 as data_loader_agent_2vs1
import pretrain.data_loader_silkmoth as data_loader_silkmoth
from pretrain.model.rnn_behavior import RNNAgent 
from pretrain.model.mlp_agent import MLPAgent 
from pretrain.utils_DQN import std_ste
from utils.utils_chase import *

# create argparse
parser = argparse.ArgumentParser()
# same as preprocessing 
parser.add_argument('--numProcess', type=int, default=16, help='No. CPUs used, default: 16')
parser.add_argument('--n_actions', type=int, default=13, help='No. of actions, default: 13') 
parser.add_argument('--reward', type=str, default='touch', help='reward type: touch (default), ...')  
# setting 
parser.add_argument('--n_agents', type=int, default=2, help='No. of agent models, default: 2')
parser.add_argument('--n_enemies', type=int, default=1, help='No. of enemies, default: 1 ')
parser.add_argument('--val_devide', type=int, default=5, help='split ratio for validation, default: 3')
parser.add_argument('--test_cond', type=str, default=None, help='test condition') 
parser.add_argument('--opponent', action='store_true', help='model opponent first, default: No')
# parser.add_argument('--score_only', action='store_true', help='use only score as a reward, default: No') 
parser.add_argument('--num_workers', type=int, default=0, help='No. of workers in dataloader, for debugging: 0 (default)')
parser.add_argument('--data_path', type=str, default='./data_result', help='data path') 
# learning
parser.add_argument('--model', type=str, default='RNN', help='learned model, default: RNN')  
parser.add_argument('--result_path', type=str, default='./data_result', help='result path') 
parser.add_argument('--env', type=str, default='animarl_agent', help='environment, default: animarl_agent')  
parser.add_argument('--batch_size', type=int, default=16, metavar='BS',help='batch size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden layer size')
parser.add_argument('--epochs', type=int, default=20, required=True, metavar='EPOC', help='train epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lmd', type=float, default=1e-5, help='lambda for L1 regularization')
parser.add_argument('--lmd2', type=float, default=10, help='lambda for constraint loss')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay of adam optimizer')
parser.add_argument('--cuda-device', default=0, type=int, metavar='N', help='which GPU to use')
parser.add_argument('--TEST', action='store_true', help='perform only test (not training), default: No')
parser.add_argument('--cont', action='store_true', help='continue learning from the saved model, default: No')
parser.add_argument("--option", type=str, default=None, help="option (default: None)")
parser.add_argument("--trainSize", type=int, default=-1, help="Size of the training sample, default: -1 (all)")
args = parser.parse_args()
numProcess = args.numProcess  
os.environ["OMP_NUM_THREADS"]=str(numProcess) 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

criterion = torch.nn.CrossEntropyLoss(reduction='none')

def display_loss(reconstruct_losses, constraint_losses, epoch, args):
    reconstruct_losses = np.array(reconstruct_losses)

    if reconstruct_losses.shape[1] == 1:
        reconstruct_losses = np.mean(reconstruct_losses) # ,0
        constraint_losses = np.mean(np.array(constraint_losses))
        # print('Epoch: {}, L_reconstruct: {:.4f}, L_norm: {:.4f}, L_constraint: {:.4f}'.format(epoch, reconstruct_losses,constraint_losses), flush=True)
        print('Epoch: {}, Outcome loss: {:.4f}'.format(epoch, reconstruct_losses), flush=True)
    elif reconstruct_losses.shape[1] <= 3:
        reconstruct_losses = np.mean(reconstruct_losses[:,:2]) # ,0
        constraint_losses = np.mean(np.array(constraint_losses)[:,:2])
        try: print('Epoch: {}, L_reconstruct: {:.4f}, L_constraint: {:.4f}'.format(epoch, reconstruct_losses,constraint_losses), flush=True)
        except: import pdb; pdb.set_trace()
        #print('Epoch: {}, L_reconstruct_1: {:.4f}, L_reconstruct_2: {:.4f}, L_reconstruct_3: {:.4f}'.format(
        #    epoch, reconstruct_losses[0], reconstruct_losses[1], reconstruct_losses[2]), flush=True)
    else:
        print('not created in neither reconstruct_losses.shape[0] == 1 nor 3')
        import pdb; pdb.set_trace()

def model_computation(inputs,model,args,epoch=-1,use_cuda=False,train=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs,state,action,terminated,filled,length = inputs
    batchSize = obs.size(0)
    n_agents = obs.size(1)
    len_time = obs.size(2)# -1
    n_actions = action.size(3)
    # state = state[:,:-1]
    terminated = terminated[:,:-1]
    mask = filled[:,:-1]
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    n_optim = 1 if not 'animarl' in model.env_name else args.n_all_agents

    if use_cuda:
        obs,state,action = obs.cuda(),state.cuda(),action.cuda()
        mask,terminated = mask.cuda(),terminated.cuda()
        length_ = length.cpu().detach().numpy()
        length = length.cuda()

    action_out = model(obs,action) # batch,time,agents,actions
    action_out = action_out[:,:-1]

    # distance based loss is difficult to backpropagate because of the non-differentiable nature
    '''action_index = torch.argmax(action_out,dim=3).unsqueeze(3) # batch,agents,time,actions

    distances = []
    for b in range(batchSize):
        state_= state[b]# .cpu().detach().numpy()
        action_index_ = action_index[b] # .cpu().detach().numpy()
        states_next = []
        for t in range(len_time-1):
            if 'agent' in model.env_name:
                action_p1 = action_index_[t,0]
                action_p2 = action_index_[t,1]
                action_e = action_index_[t,2]           
                state_p1, state_p2, state_e = transition_agent(state_[t], action_p1, action_p2, action_e, args.params)
                states_next.append(torch.stack([state_p1,state_p2,state_e]))

        states_next = torch.stack(states_next) 
        distances_ = []
        for n in range(n_agents):               
            traj_expert = state_.reshape(-1,n_agents,args.state_dim)[1:int(length_[b]),n,:2]
            traj_agent = states_next[:int(length_[b])-1,n,0,:2] # time,agents,,dim,xy
            # Euclid distance is enough because traj_agent and traj_expert are the same length
            distance = torch.sqrt(torch.sum((traj_agent - traj_expert) ** 2, 1) + 1e-6)
            # dtw, distance_matrix_ = warping_paths_fast(traj_agent.astype(np.float64),traj_expert.astype(np.float64)) # [:,n,:self.space_dim].astype(np.float64), traj_expert[n,:,:self.space_dim].astype(np.float64))
            # reward = -(distance_matrix_[1:,1:]).min(axis=1)
            distances_.append(torch.mean(distance))
        distances.append(torch.stack(distances_))

    distances = torch.stack(distances)
    distances = torch.mean(distances, 0)
    reconstruct_error = distances'''

    # action based reconstruct_error 
    # reconstruct_error = action_out - action[:,:,1:].permute(0,2,1,3) # :-1
    action_GT = action[:,:,1:].permute(0,2,1,3).argmax(dim=3).long() # batch,agents,time
    action_out = action_out.permute(0,3,2,1) # batch,actions,agents,time
    try: mask_ = mask.permute(0,2,1).unsqueeze(1).expand_as(action_out) 
    except: import pdb; pdb.set_trace()
    # masked_rec_error = reconstruct_error * mask_

    # constraint loss
    # if args.env == "animarl_silkmoth":
    #    constraint_error
    
    constraint_error = torch.zeros_like(action_out).to(device) * mask_

    if False: #n_optim == 1:
        reconstruct_loss = (masked_rec_error ** 2).sum() / mask.sum() 
        constraint_loss = (constraint_error).sum() / mask.sum() 
    else:
        reconstruct_loss,constraint_loss = [],[]
        for n in range(n_optim):
            try: 
                loss = criterion(action_out[:,:,n] * mask_[:,:,n], action_GT[:,:,n] * mask_[:,0,n].long())
            except: 
                import pdb; pdb.set_trace()
            loss_sum = loss.masked_fill(mask_[:,0,n].bool(), 0).sum()
            reconstruct_loss.append(loss_sum / mask_.sum())
            # reconstruct_loss.append((masked_rec_error[:,:,n] ** 2).sum() / mask[:,:,n].sum())
            constraint_loss.append(constraint_error[:,:,n].sum() / mask[:,:,n].sum())
    #if epoch == 19:
    #    import pdb; pdb.set_trace()

    if train:
        return reconstruct_loss, constraint_loss
    else:
        return reconstruct_loss, constraint_loss, action_out, length

def trainmodel(train_loader, val_loader,test_loader, model, args, epochs, optimizer, criterion, 
                  use_cuda=False, save_model=None, TEST=False):

    # Train network
    best_loss_val = float('inf') # torch.tensor(float('inf')).to(device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_optim = 1 if not 'animarl' in model.env_name else args.n_all_agents
 
    no_update = 0 
    if not TEST:
        # train # model.train()
        for epoch in range(epochs):
            reconstruct_losses,constraint_losses = [],[]

            for obs,state,action,terminated,filled,length in tqdm(train_loader):
                
                inputs = [obs,state,action,terminated,filled,length]

                reconstruct_loss,constraint_loss = model_computation(inputs,model,args,epoch,use_cuda=use_cuda)

                if False: # n_optim == 1:
                    loss = reconstruct_loss + args.lmd * constraint_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.params,20) # model.parameters(), 20)
                    optimizer.step()
                    reconstruct_losses.append(reconstruct_loss.item())
                    constraint_losses.append(constraint_loss.item())
                else:
                    loss = [[] for _ in range(n_optim)]
                    for n in range(n_optim):
                        loss[n] = reconstruct_loss[n].clone() + args.lmd * constraint_loss[n].clone()
                        optimizer[n].zero_grad()
                        loss[n].backward(retain_graph=True)
                    # torch.stack(loss).sum().backward()
                    for n in range(n_optim):
                        torch.nn.utils.clip_grad_norm_(model.params[n],20) 
                        optimizer[n].step()
                    reconstruct_losses.append(torch.stack(reconstruct_loss).to('cpu').detach().numpy().copy())
                    constraint_losses.append(torch.stack(constraint_loss).to('cpu').detach().numpy().copy())
            
            display_loss(reconstruct_losses, constraint_losses, epoch, args)    

            # validation
            print('Validation:')
            loss_val = model_eval(model, val_loader, args, epoch, eval_use_cuda=use_cuda)
            try: 
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    no_update = 0
                    if save_model:
                        print('Best model. Saving...\n')
                        torch.save(model.state_dict(), save_model+'agent.th')
                        if False: # n_optim == 1:
                            torch.save(optimizer.state_dict(), save_model+'opt.th')
                        else:
                            for n in range(n_optim):
                                torch.save(optimizer[n].state_dict(), save_model+'opt_'+str(n)+'.th')

                elif np.isnan(loss_val) or (epoch==0 and loss_val == best_loss_val):
                    print('loss is nan or inf')
                    import pdb; pdb.set_trace()
                else:
                    no_update += 1
                    if no_update >= 3:
                        model, optimizer = load_best_model(save_model,args,model,optimizer)    
                        print('since no update continues, best model was loaded')   
                        no_update = 0

            except: import pdb; pdb.set_trace()
    else:
        epoch = 0
                
    print('Test:')
    model, optimizer = load_best_model(save_model,args,model,optimizer)
    loss_test = model_eval(model, test_loader, args, epoch, eval_use_cuda=use_cuda, save=True, TEST=True)
    


def model_eval(model, dataloader, args, epoch, eval_use_cuda=False, save=False, TEST=False):

    outcomes, inputs = transfer_data(model, dataloader, args, epoch, eval_use_cuda, save=save, TEST=TEST)
    
    actions, reconstruct_loss, constraint_losses, loss_all = outcomes
    states, lengths = inputs

    n_agents = args.n_agents
    std = False 

    if False:
        # when extra evaluation is needed
        extra_result = []
    else:
        extra_result = []
        print('L_reconstruct: {:.4f} (+/-) {:.4f}, L_constraint: {:.4f} (+/-) {:.4f}'.format(
            np.mean(reconstruct_loss),std_ste(reconstruct_loss,std),np.mean(constraint_losses),std_ste(constraint_losses,std)), flush=True)

    if save:
        res = [outcomes, inputs, extra_result]
        with open(args.save_results+'.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=4) 
        print(args.save_results+'.pkl is saved')
        import pdb; pdb.set_trace()
    return np.mean(loss_all)

def transfer_data(model, dataloader, args, epoch, eval_use_cuda=False, save=False, TEST=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_optim = 1 if not 'animarl' in model.env_name else args.n_all_agents
    print('TEST ='+str(TEST))

    with torch.no_grad():
        # model.eval()
        Qs = []
        Qs_local = []
        loss_all = []
        reconstruct_losses,constraint_losses = [],[]
        actions, lengths, states = [],[],[]

        for obs,state,action,terminated,filled,length in dataloader:
            inputs = [obs,state,action,terminated,filled,length]
            reconstruct_loss,constraint_loss, action, length = model_computation(
                inputs,model,args,use_cuda=eval_use_cuda,train=False)

            if False: #n_optim == 1:
                loss = reconstruct_loss + args.lmd * constraint_loss
                reconstruct_losses.append(np.array(reconstruct_loss.item())[np.newaxis])
                loss_all.append(np.array(loss)[np.newaxis])
                constraint_losses.append(np.array(constraint_loss.item())[np.newaxis])
            else:
                loss = [[] for _ in range(n_optim)]
                for n in range(n_optim):
                    loss[n] = reconstruct_loss[n] + args.lmd * constraint_loss[n]

                reconstruct_losses.append(torch.stack(reconstruct_loss).to('cpu').detach().numpy().copy())
                loss_all.append(torch.stack(loss).to('cpu').detach().numpy().copy())
                constraint_losses.append(torch.stack(constraint_loss).to('cpu').detach().numpy().copy())

            '''if args.env == "animarl_silkmoth":
                cos = action[:,:,:,3]
                sin = action[:,:,:,2]
                theta_ = torch.atan2(sin,cos)
                theta_[theta_<0] += 2*math.pi
                action_ = torch.cat([action[:,:,:,:2],theta_.unsqueeze(3)],3)
                action = action_'''
 
            # detach
            action = detach(action,eval_use_cuda)
            length = detach(length,eval_use_cuda)

            # append            
            actions.append(action) # batch,agents,time,actions
            states.append(state)
            lengths.append(length)
            
        try: display_loss(reconstruct_losses, constraint_losses, epoch, args)
        except: import pdb; pdb.set_trace()
        # concatenate
        actions = np.concatenate(actions)
        states = np.concatenate(states)
        lengths = np.concatenate(lengths)
        loss_all = np.concatenate(loss_all)
        reconstruct_losses = np.concatenate(reconstruct_losses)
        constraint_losses = np.concatenate(constraint_losses)

        # for saving
        outcomes = [actions, reconstruct_losses, constraint_losses, loss_all]
        inputs = [states, lengths]
        return outcomes, inputs

def detach(data,eval_use_cuda):
    if eval_use_cuda:
        return data.to('cpu').detach().data.numpy()
    else:
        return data.detach().data.numpy()

def load_best_model(save_model,args,model, optimizer):
    n_optim = 1 if not 'animarl' in model.env_name else args.n_all_agents
    # weight_before = model.fc1[0].weight.clone()
    model.load_state_dict(torch.load("{}agent.th".format(save_model), map_location=lambda storage, loc: storage))
    # weight_after = model.fc1[0].weight
    if False: #n_optim == 1:
        optimizer.load_state_dict(torch.load("{}opt.th".format(save_model), map_location=lambda storage, loc: storage))
    else:
        for n in range(n_optim):
            optimizer[n].load_state_dict(torch.load("{}opt_{}.th".format(save_model,n), map_location=lambda storage, loc: storage))

    return model, optimizer

if __name__ == '__main__':
    # constants
    args.Fs = 0.1
    if args.env == "animarl_agent" or "fly" in args.env or "newt" in args.env:
        args.n_agents = 2
        args.n_enemies = 1
        args.reward = 'touch'  
        args.state_shape = 18
    elif args.env == "animarl_silkmoth":
        args.n_agents = 1
        args.n_enemies = 0
        args.reward = 'reach'
        args.state_shape = 14
    
    n_agents = args.n_agents
    seed = 0
    CUDA = True
    args.behavior = True 
    args.transition = False

    # parameters
    n_actions = args.n_actions
    val_devide = args.val_devide

    args.n_all_agents = args.n_agents + args.n_enemies
    n_feature = args.state_shape 
    
    args.n_agents_team = args.n_agents
    args.input_data_path = os.path.join(args.result_path,'preprocessed')

    # random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.set_device(args.cuda_device)
        torch.cuda.manual_seed(seed)

    if args.env != "animarl_silkmoth":
        args.env += '_' + str(n_agents) + 'vs' + str(args.n_enemies)

    if args.env == "animarl_agent_2vs1":
        # input_data_path = '/home/fujii/workspace3/work/tag/analysis/cae/npy/'  
        args.state_dim = 4
        args.obs_shape = 19
        n_files = 8
        args.max_len = 148 # 42 # 302
        data_loader = data_loader_agent_2vs1
        all_indices = np.arange(n_files*100)

        preprocessed = np.load(os.path.join(args.input_data_path,'agent'+ '_'+ str(args.trainSize)+'.npz'), allow_pickle=True)
        us = preprocessed['arr_5']
        ds = preprocessed['arr_6']
        args.params = {
            'mass_p1': 1,
            'speed_p1': us[0],
            'damping_p1': ds[0],
            'dt': args.Fs,
            'mass_p2': 1,
            'speed_p2': us[1],
            'damping_p2': ds[1],
            'mass_e': 1,
            'speed_e': us[2],
            'damping_e': ds[2],
            'n_mate': n_agents,
            'n_adv': args.n_enemies
        }
    elif args.env == "animarl_fly_2vs1":
        args.state_dim = 4
        args.obs_shape = 19
        n_files = 107 # 114
        args.max_len = 480
        data_loader = data_loader_agent_2vs1
        all_indices = np.arange(n_files)

        preprocessed = np.load(os.path.join(args.input_data_path,'fly'+ '_'+ str(args.trainSize)+'.npz'), allow_pickle=True)
        us = preprocessed['arr_5']
        ds = preprocessed['arr_6']
        args.params = {
            'mass_p1': 1,
            'speed_p1': us[0],
            'damping_p1': ds[0],
            'dt': args.Fs,
            'mass_p2': 1,
            'speed_p2': us[1],
            'damping_p2': ds[1],
            'mass_e': 1,
            'speed_e': us[2],
            'damping_e': ds[2],
            'n_mate': n_agents,
            'n_adv': args.n_enemies
        }

    elif args.env == "animarl_newt_2vs1":
        args.state_dim = 4
        args.obs_shape = 18
        n_files = 447
        args.max_len = 500
        data_loader = data_loader_agent_2vs1
        all_indices = np.arange(n_files)

        preprocessed = np.load(os.path.join(args.input_data_path,'newt'+ '_'+ str(args.trainSize)+'.npz'), allow_pickle=True)
        us = preprocessed['arr_5']
        ds = preprocessed['arr_6']
        args.params = {
            'mass_p1': 1,
            'speed_p1': us[0],
            'damping_p1': ds[0],
            'dt': args.Fs,
            'mass_p2': 1,
            'speed_p2': us[1],
            'damping_p2': ds[1],
            'mass_e': 1,
            'speed_e': us[2],
            'damping_e': ds[2],
            'n_mate': n_agents,
            'n_adv': args.n_enemies
        }

    elif args.env == "animarl_silkmoth":
        args.state_dim = 13
        args.obs_shape = 11
        args.Fs = 1/2
        args.max_len = 300*int(1/args.Fs)-1 # np.inf 
        data_loader = data_loader_silkmoth
        if args.option == 'left':
            preprocessed = np.load(os.path.join(args.input_data_path,'silkmoth_left'+ '_'+ str(args.trainSize)+'.npz'), allow_pickle=True)
            args.origin = np.array([-1,0])
        else:
            preprocessed = np.load(os.path.join(args.input_data_path,'silkmoth'+ '_'+ str(args.trainSize)+'.npz'), allow_pickle=True)
            args.origin = np.array([0,0])        # states = preprocessed['arr_0']
        args.n_folders = 10
        args.n_files = 6 # states.shape[0]
        all_indices = np.arange(args.n_folders*args.n_files)
        us = preprocessed['arr_6']
        ds = preprocessed['arr_7']
        args.params = {
            'mass': 1,
            'speed': us[0],
            'damping': ds[0],
            'dt': args.Fs,
            'n_mate': args.n_agents,
            'n_adv': args.n_enemies,
            'origin': args.origin
        }

        '''input_data_path = args.data_path+'/silkmoth/VRdata/'  
        n_folders = 8
        n_files = 30
        args.Fs = 1/30

        indices = np.arange(n_files)
        args.n_files = n_files
        args.n_folders = n_folders
        if False:
            foldernames = ['Cond1','Cond2','Cond3','Cond4','Cond5','Condi','Condii','Condiii']
            filenames = ['cond1','cond2','cond3','cond4','cond5','condi','condii','condiii']
            lengths = []
            for i in range(n_folders):
                for j in range(n_files):
                    filename = input_data_path+'/'+foldernames[i]+'/'+filenames[i]+'_'+str(j+1)+'.csv'
                    df = pd.read_csv(filename)
                    pos = df.values[:,1:3]
                    end_dist = np.sqrt(np.sum(pos[-1]**2))
                    print('folder: '+str(i)+', file: '+str(j+1)+', length: '+str(len(df))+', end_dist: '+str(int(end_dist)))
                    lengths.append(len(df))'''

    # args.input_data_path = args.result_path
    args.data = args.env
    args.state_shape = args.n_all_agents*args.state_dim
    n_feature = args.obs_shape + n_actions

    # train/valid/test split
    if args.env == "animarl_agent_2vs1":
        if True: # args.option == 'CF':
            train_ids = np.concatenate([all_indices[:200],all_indices[379:579]])
            val_ids = np.concatenate([all_indices[200:225],all_indices[579:604]])
            test_ids = np.concatenate([all_indices[286:311],all_indices[663:688]])
        else:
            train_ids = all_indices[:args.trainSize]
            val_ids = all_indices[400:450] 
            test_ids = all_indices[450:500] 
    elif "animarl_fly" in args.env:
        train_ids = all_indices[:84]
        val_ids = all_indices[84:97] # [84:104]
        test_ids = all_indices[97:107] # [104:114]
    elif "animarl_newt" in args.env:
        train_ids = all_indices[:args.trainSize]
        val_ids = all_indices[240:260]
        test_ids = all_indices[260:280]
    elif args.test_cond is None and "animarl_silkmoth" in args.env:
        test_ids = np.concatenate([all_indices[24:30],all_indices[54:60]],0) # movie 5
        train_val_ids = np.concatenate([all_indices[:24],all_indices[30:54]],0) # movie 1-4
        train_ids, val_ids,_,_ = train_test_split(train_val_ids, train_val_ids, test_size=1/(val_devide-1), random_state=seed)  
    elif args.test_cond is None: 
        train_ids, test_ids,_,_ = train_test_split(all_indices, all_indices, test_size=1/val_devide, random_state=seed)
        train_ids, val_ids,_,_ = train_test_split(train_ids, train_ids, test_size=1/(val_devide-1), random_state=seed)  
    elif args.env == "animarl_silkmoth" and (args.test_cond == 'cont_odor' or args.test_cond == 'odor_cont'):
        if args.test_cond == 'cont_odor':
            train_id, val_id,_,_ = train_test_split(all_indices[:30], all_indices[:30], test_size=1/val_devide, random_state=seed)
            test_ids = all_indices[30:]
        elif args.test_cond == 'odor_cont':
            train_id, val_id,_,_ = train_test_split(all_indices[30:], all_indices[30:], test_size=1/val_devide, random_state=seed)
            test_ids = all_indices[:30]
    else:
        import pdb; pdb.set_trace()

    print('train: '+str(len(train_ids))+', val: '+str(len(val_ids))+', test: '+str(len(test_ids))+' sequences')  

    # Datasets
    train_dataset = data_loader.Dataset(train_ids, args, TEST = False)
    val_dataset = data_loader.Dataset(val_ids, args, TEST = False)
    test_dataset = data_loader.Dataset(test_ids, args, TEST = True)

    # DataLoaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if CUDA else {} #  {} # 
    # mp.set_start_method('spawn')
    shuffle = True
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=shuffle, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False, **kwargs)

    # model and optimizer
    args.model_variant = args.model
    if 'RNN' in args.model:
        args.rnn_hidden_dim = args.hidden_size
        model = RNNAgent(n_feature, args)
    elif 'MLP' in args.model:
        model = MLPAgent(n_feature, args)

    params = list(model.parameters())

    if not "animarl" in model.env_name:
        optimizer = optim.Adam(model.params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = [optim.Adam(model.params[n], lr=args.lr, weight_decay=args.weight_decay) for n in range(args.n_all_agents)]

    # str_param = ''
    func = 'BC'
    str_param = '_'+args.reward if args.test_cond is None else '_'+args.reward+'_'+args.test_cond
    str_param += '_'+args.option if args.option is not None else ''

    args.save_model = os.path.join(args.result_path, 'model_'+func, args.model+str_param+'-'+args.data+'-ts-'+str(args.trainSize)+'-lr-'+str(args.lr)) + os.sep
    args.save_results = os.path.join(args.result_path, 'results_'+func, args.model+str_param+'-'+args.data+'-ts-'+str(args.trainSize)+'-lr-'+str(args.lr)) + os.sep
    #args.save_model = args.result_path+'/model_behavior'+'/'+args.model+str_param+'-'+args.data+'/' # ../AniMARL_results
    #args.save_results = args.result_path+'/results_behavior'+'/'+args.model+str_param+'-'+args.data+'/'
    if not os.path.isdir(args.save_model): # setting.OUTPUT_PATH+'/model_Q'):
        os.makedirs(args.save_model)
    if not os.path.isdir(args.save_results):
        os.makedirs(args.save_results)
    print("save_model ==> ", args.save_model, 'agent.th')
    print("save_results ==> ", args.save_results)
    

    if args.cont:
        print('args.cont = True')
        if os.path.exists(args.save_model+'agent.th'): # .pt'): 
            model, optimizer = load_best_model(args.save_model,args,model,optimizer)
            print('best model was loaded')   
            import pdb; pdb.set_trace()
        else:
            print('args.cont = True but file did not exist')

    if CUDA: # use_cuda:
        print("====> Using CUDA device: ", torch.cuda.current_device(), flush=True)
        if 'RNN' in args.model:
            #model.cuda()
            # model = model.to('cuda')
            for n in range(args.n_all_agents):
                model.fc_in[n].cuda() # fc1
                model.rnn[n].cuda() # rnn
                model.fc_out[n].cuda() # fc2
        elif 'MLP' in args.model:
            for n in range(args.n_all_agents):
                model.fc_in[n].cuda() # fc1
                model.fc_state[n].cuda() # 
                model.fc_advantage[n].cuda() # fc2
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
    # training and test
    trainmodel(train_loader, val_loader, test_loader, model, args, epochs= args.epochs,
                  criterion=F.mse_loss, optimizer=optimizer,
                  use_cuda=CUDA, save_model=args.save_model, TEST=args.TEST)
