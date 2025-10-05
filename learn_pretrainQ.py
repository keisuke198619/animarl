import os, sys, glob, argparse, random, copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import multiprocessing as mp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from dtaidistance.dtw_ndim import warping_paths_fast

import pretrain.data_loader_agent_2vs1 as data_loader_agent_2vs1
import pretrain.data_loader_silkmoth as data_loader_silkmoth
from pretrain.model.rnn_agent import RNNAgent 
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
parser.add_argument('--result_path', type=str, default='./data_result', help='result path') 
parser.add_argument('--model', type=str, default='DQN', help='learned model, default: DQN')  
parser.add_argument('--env', type=str, default='animarl_agent', help='environment, default: animarl_agent')  
parser.add_argument('--batch_size', type=int, default=16, metavar='BS',help='batch size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden layer size')
parser.add_argument('--epochs', type=int, default=20, required=True, metavar='EPOC', help='train epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma for Q learning and SARSA')
parser.add_argument('--lmd', type=float, default=1e-5, help='lambda for L1 regularization')
parser.add_argument('--lmd2', type=float, default=50, help='lambda for supervised loss')
parser.add_argument('--lmd_TD', type=float, default=1.0, help='lambda for TD loss')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay of adam optimizer')
parser.add_argument('--cuda-device', default=0, type=int, metavar='N', help='which GPU to use')
parser.add_argument('--AS', action='store_true', help='action crossentropy loss for imitation, default: No')
parser.add_argument('--DIL', action='store_true', help='distance-based imitation learning, default: No')
parser.add_argument('--TEST', action='store_true', help='perform only test (not training), default: No')
parser.add_argument('--cont', action='store_true', help='continue learning from the saved model, default: No')
parser.add_argument("--option", type=str, default=None, help="option (default: None)")
parser.add_argument("--trainSize", type=int, default=-1, help="Size of the training sample, default: -1 (all)")
args = parser.parse_args()
numProcess = args.numProcess  
os.environ["OMP_NUM_THREADS"]=str(numProcess) 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

print(args)

NLLlogsoftmax = True
if NLLlogsoftmax:
    supervisedLoss = [torch.nn.NLLLoss(ignore_index=-100) for _ in range(args.n_agents+args.n_enemies)]
else:
    supervisedLoss = [torch.nn.CrossEntropyLoss(ignore_index=-100) for _ in range(args.n_agents+args.n_enemies)]

def display_loss(outcome_losses, norm_losses, supervised_losses, epoch, args):
    outcome_losses = np.array(outcome_losses)

    if outcome_losses.shape[1] == 1:
        outcome_losses = np.mean(outcome_losses) # ,0
        norm_losses = np.mean(np.array(norm_losses))
        supervised_losses = np.mean(np.array(supervised_losses))
        # print('Epoch: {}, L_outcome: {:.4f}, L_norm: {:.4f}, L_supervised: {:.4f}'.format(epoch, outcome_losses,norm_losses,supervised_losses), flush=True)
        print('Epoch: {}, Outcome loss: {:.4f}'.format(epoch, outcome_losses), flush=True)
    elif outcome_losses.shape[1] <= 3:
        outcome_losses = np.mean(outcome_losses[:,:2]) # ,0
        norm_losses = np.mean(np.array(norm_losses)[:,:2])
        supervised_losses = np.mean(np.array(supervised_losses)[:,:2])
        if args.DIL:
            print('Epoch: {}, L_outcome: {:.4f}, L_norm: {:.4f}, distance (reward): {:.4f}'.format(epoch, outcome_losses,norm_losses,supervised_losses), flush=True)
        else:
            print('Epoch: {}, L_outcome: {:.4f}, L_norm: {:.4f}, L_supervised: {:.4f}'.format(epoch, outcome_losses,norm_losses,supervised_losses), flush=True)

        #print('Epoch: {}, L_outcome_1: {:.4f}, L_outcome_2: {:.4f}, L_outcome_3: {:.4f}'.format(
        #    epoch, outcome_losses[0], outcome_losses[1], outcome_losses[2]), flush=True)
    else:
        print('not created in neither outcome_losses.shape[0] == 1 nor 3')
        import pdb; pdb.set_trace()

def get_submodules(module):
  for name, submodule in module.named_modules():
    if not isinstance(submodule, nn.ModuleList):
      yield submodule
    else:
      yield from get_submodules(submodule)

def model_computation(inputs,model,args,epoch=-1,use_cuda=False,train=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs,state,action,avail_action,reward,terminated,filled,length = inputs
    batchSize = obs.size(0)
    n_agents = obs.size(1)
    len_time = obs.size(2)# -1
    n_actions = action.size(3)
    avail_action = avail_action.permute(0,2,1,3) # [:,:,:-1]
    # state = state[:,:-1]
    terminated = terminated[:,:-1]
    mask = filled[:,:-1]
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    n_optim = args.n_all_agents

    if use_cuda:
        obs,state,action,reward = obs.cuda(),state.cuda(),action.cuda(),reward.cuda()
        avail_action,mask,terminated = avail_action.cuda(),mask.cuda(),terminated.cuda()
        length_ = length.cpu().detach().numpy()
        length = length.cuda()
    # rewards = reward[:,:1,None].repeat(1,len_time-1,1)
    # rewards = reward[:,None].repeat(1,len_time-1,1)
    # if 'animarl' in model.env_name:
    #    rewards[:,:-1,:] = 0
    rewards = torch.zeros(batchSize,len_time-1,n_agents).to(device)
    if args.DIL: 
        no_direct_reward = True
    else:
        no_direct_reward = False
    if 'animarl' in model.env_name:
        for b in range(batchSize):
            if not no_direct_reward:
                rewards[b,int(length_[b])-1,:] = reward[b]
            else:
                rewards[b,int(length_[b])-1,:] = reward[b]*0 # *1e-10
    else:
        import pdb; pdb.set_trace()
    
    # qs = model(obs, action) # qs: batch,time,agents,actions
    qs,out_cond = model(obs, action) # qs: batch,time,agents,actions
    
    action_index = torch.argmax(action[:,:,:-1],dim=3).unsqueeze(3).permute(0,2,1,3)
    # torch.where(action[:,:,:-1]==1)[0].reshape(batchSize,n_agents,len_time-1,1).permute(0,2,1,3)

    if args.option == 'CF':
        cond_gt = obs[:,:,:,-1]
        # https://github.com/keisuke198619/TGV-CRN/blob/main/train_MADSW.py
        loss_cond = [torch.zeros(1).to(device) for _ in range(n_agents)]
        for n in range(n_optim):
            for b in range(batchSize):
                length_b = int(length_[b])
                loss_cond[n] += F.cross_entropy(out_cond[b, n, :length_b, :].reshape(-1, out_cond.size(-1)), 
                        cond_gt[b, n, :length_b].reshape(-1).long())
            loss_cond[n] /= batchSize
        

    
    if args.DIL:
        distances = []
        for b in range(batchSize):
            state_= state[b].cpu().detach().numpy()
            if 'animarl_agent' in model.env_name: # and args.option == 'CF' or 'animarl_silkmoth' in model.env_name):
                state_ = state_[:,:-1]
            action_index_ = action_index[b].cpu().detach().numpy()
            states_next = []
            for t in range(len_time-1):
                if 'agent' in model.env_name or 'fly' in model.env_name or 'newt' in model.env_name:
                    action_p1 = action_index_[t,0]
                    action_p2 = action_index_[t,1]
                    action_e = action_index_[t,2]           
                    state_p1, state_p2, state_e, _,_,_ = transition_agent(state_[t], action_p1, action_p2, action_e, args.params)
                    states_next.append(np.stack([state_p1,state_p2,state_e]))
                elif 'silkmoth' in model.env_name:
                    action_ = action_index_[t,0]         
                    state_next_, _ = transition_single(state_[t], action_, args.params)
                    states_next.append(np.stack([state_next_]))

            states_next = np.stack(states_next) 
            if 'silkmoth' in model.env_name:
                states_next = states_next[:,:,:,None,:]
            distances_ = []
            for n in range(n_agents):               
                try: 
                    traj_expert = state_.reshape(-1,n_agents,args.state_dim)[1:int(length_[b]),n,:2]
                    traj_agent = states_next[:int(length_[b])-1,n,0,0,:2] # time,agents,dim,1,xy
                except: import pdb; pdb.set_trace()
                # Euclid distance is enough because traj_agent and traj_expert are the same length
                traj_agent.astype(np.float64),traj_expert.astype(np.float64)
                distance = torch.from_numpy(np.sqrt(np.sum((traj_agent - traj_expert) ** 2, 1) + 1e-6))
                # dtw, distance_matrix_ = warping_paths_fast(traj_agent.astype(np.float64),traj_expert.astype(np.float64)) # [:,n,:self.space_dim].astype(np.float64), traj_expert[n,:,:self.space_dim].astype(np.float64))
                # reward = -(distance_matrix_[1:,1:]).min(axis=1)
                if use_cuda:
                    distance = distance.cuda()
                distances_.append(torch.mean(distance))
                rewards[b,1:int(length_[b]),n] += args.lmd2 * distance # reward
            distances.append(distances_)

        distances = torch.tensor(distances)  
        distances = torch.mean(distances, 0)

    qs_ = qs[:,:-1]
    # Pick the Q-Values for the actions taken by each agent
    chosen_action_qvals_ = torch.gather(qs_, dim=3, index=action_index).squeeze(3)  # why NG?
    # chosen_action_qvals = torch.gather(qs_[13], dim=2, index=action_index[13]) # why NG?

    x_qs = qs.clone().detach()
    x_qs[avail_action == 0] = -9999999
    max_action_qvals, max_action_index = x_qs[:, :-1].max(dim=3) # batch_size,time_limit,n_agents

    max_action_index = max_action_index.detach().unsqueeze(3)
    is_max_action = (max_action_index == action_index).int().float()

    if "DQN" in args.model:
        # Calculate the Q-Values necessary for the target
        target_qs = qs.clone()
        target_qs = target_qs[:, 1:]

        # Max over target Q-Values
        mac_out_detach = qs.clone().detach()
        
        mac_out_detach[avail_action == 0] = -9999999
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        
        target_max_qvals = torch.gather(target_qs, 3, cur_max_actions).squeeze(3) # why OK?

        # Calculate 1-step Q-Learning targets
        targets = rewards + args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals_ - targets.detach())

        # mask_ = mask[:,:,0,None] # mask.expand_as(td_error) 

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # norm_loss = F.l1_loss(qs, target=torch.zeros_like(qs), reduction='none')[:, :-1]
        # mask_expand = mask.unsqueeze(-1).expand_as(norm_loss)

        # supervised loss
        if args.AS:
            if NLLlogsoftmax:
                softmax = F.log_softmax(qs_.clone(),dim=3)
            else: 
                softmax = F.softmax(qs_.clone(),dim=3)
            action_expert = action_index.clone() # action[:,:,:-1].clone().permute(0,2,1,3).long()
            action_expert[mask==0] = -100

        elif not args.DIL: # DQfD
            QE = chosen_action_qvals_
            margin_value = 0.8
            A = max_action_index
            #self.vc.calcQ(self.vc.predictNet, s, aE)
            # A1, A2 = np.array(A)[:2]  # action with largest and second largest Q
            #maxA = A2 if (A1 == aE).all() else A1
            #Q = self.vc.calcQ(self.vc.predictNet, s, maxA)
            margin = torch.zeros(A.shape).to(device) 
            margin[A!=action_index] = margin_value 
            # mask_ = mask.unsqueeze(3)#.expand_as(QE)
            supervised_loss = ((torch.max(max_action_qvals.unsqueeze(3) + margin) - QE)*mask)# .sum(dim=(0,1)) / mask.sum(dim=(0,1)) # torch.max(Q + margin) - QE[aE][0]


        if False: # n_optim == 1:
            outcome_loss = (masked_td_error ** 2).sum() / mask.sum()
            import pdb; pdb.set_trace()
            norm_loss_ = (norm_loss * mask_expand).sum() / mask_expand.sum()
            supervised_loss_ = supervisedLoss(softmax.reshape(-1,n_actions),action_expert.reshape(-1,))
        else:
            outcome_loss,norm_loss_,supervised_loss_ = [],[],[]
            norm_loss = [[] for _ in range(n_optim)]
            n_ = 0
            for name, layer in model.named_modules():
                if isinstance(layer, nn.ModuleList):
                    for j, paras in enumerate(layer.parameters()):
                        if paras.requires_grad:
                            for n in range(n_optim): 
                                if n_ == 0:
                                    norm_loss[n] = paras.abs().sum()
                                else:
                                    norm_loss[n] += paras.abs().sum()
                            n_ += 1
            
            '''for paras in model.parameters():
                if paras.requires_grad:
                    for para in paras:
                        
                        try: norm_loss_.append(para.abs().sum())
                        except: import pdb; pdb.set_trace()     '''

            for n in range(n_optim):
                outcome_loss.append((masked_td_error[:,:,n] ** 2).sum() / mask[:,:,n].sum())
                norm_loss_.append(norm_loss[n]/args.total_params)
                # norm_loss_.append((norm_loss[:,:,n] * mask_expand[:,:,n]).sum() / mask_expand[:,:,n].sum())
                if args.AS:
                    supervised_loss_.append(supervisedLoss[n](softmax[:,:,n].reshape(-1,n_actions),action_expert[:,:,n].reshape(-1,)))
                elif args.DIL and args.option == 'CF':
                    supervised_loss_.append(loss_cond[n])
                elif args.DIL:
                    supervised_loss_.append(distances.to(device))
                else: # DQfD
                    supervised_loss_.append(supervised_loss[:,:,n].sum() / mask[:,:,n].sum())
    else:
        import pdb; pdb.set_trace()

    if train:
        return outcome_loss,norm_loss_,supervised_loss_ 
    else:
        return outcome_loss, norm_loss_,supervised_loss_, qs, action, reward, length

def trainmodel(train_loader, val_loader,test_loader, model, args, epochs, optimizer, criterion, 
                  use_cuda=False, save_model=None, TEST=False):

    # Train network
    best_loss_val = float('inf') # torch.tensor(float('inf')).to(device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_optim = args.n_all_agents
 
    no_update = 0 
    if not TEST:
        # train # model.train()
        for epoch in range(epochs):
            outcome_losses,norm_losses,supervised_losses = [],[],[]
            for obs,state,action,avail_action,reward,terminated,filled,length in tqdm(train_loader):
                
                inputs = [obs,state,action,avail_action,reward,terminated,filled,length]

                outcome_loss,norm_loss,supervised_loss = model_computation(inputs,model,args,epoch,use_cuda=use_cuda)

                if False: # n_optim == 1:
                    if args.DIL:
                        loss = args.lmd_TD * outcome_loss + args.lmd * norm_loss
                    if "DQN" in args.model:
                        loss = args.lmd_TD * outcome_loss + args.lmd * norm_loss + args.lmd2 * supervised_loss
                    else: 
                        loss = outcome_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.params,20) # model.parameters(), 20)
                    optimizer.step()
                    outcome_losses.append(outcome_loss.item())
                    norm_losses.append(norm_loss.item())
                    supervised_losses.append(supervised_loss.item())
                else:
                    loss = [[] for _ in range(n_optim)]
                    for n in range(n_optim):
                        if args.DIL and args.option == 'CF':
                            loss[n] = args.lmd_TD * outcome_loss[n].clone() + args.lmd * norm_loss[n].clone() + args.lmd2 * supervised_loss[n].clone()
                        if args.DIL:
                            loss[n] = args.lmd_TD * outcome_loss[n].clone() + args.lmd * norm_loss[n].clone() 
                        elif "DQN" in args.model:
                            loss[n] = args.lmd_TD * outcome_loss[n].clone() + args.lmd * norm_loss[n].clone() + args.lmd2 * supervised_loss[n].clone()
                        else: 
                            loss[n] = outcome_loss[n].clone()
                        optimizer[n].zero_grad()
                        loss[n].backward(retain_graph=True)
                    # torch.stack(loss).sum().backward()
                    for n in range(n_optim):
                        torch.nn.utils.clip_grad_norm_(model.params[n],20) 
                        optimizer[n].step()
                    outcome_losses.append(torch.stack(outcome_loss).to('cpu').detach().numpy().copy())
                    norm_losses.append(torch.stack(norm_loss).to('cpu').detach().numpy().copy())
                    supervised_losses.append(torch.stack(supervised_loss).to('cpu').detach().numpy().copy())
            display_loss(outcome_losses, norm_losses, supervised_losses, epoch, args)    

            # validation
            print('Validation:')
            loss_val = model_eval(model, val_loader, args, epoch, eval_use_cuda=use_cuda)
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                no_update = 0
                if save_model:
                    print('Best model. Saving...\n')
                    torch.save(model.state_dict(), save_model+'agent.th')
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
    else:
        epoch = 0
                
    print('Test:')
    model, optimizer = load_best_model(save_model,args,model,optimizer)
    loss_test = model_eval(model, test_loader, args, epoch, eval_use_cuda=use_cuda, save=True, TEST=True)
    


def model_eval(model, dataloader, args, epoch, eval_use_cuda=False, save=False, TEST=False):

    outcomes, inputs = transfer_data(model, dataloader, args, epoch, eval_use_cuda, save=save, TEST=TEST)
    
    Qs, outcome_loss, norm_losses, supervised_losses, loss_all = outcomes
    states, actions, rewards, lengths = inputs

    n_agents = args.n_agents
    std = False 

    if False:
        # when extra evaluation is needed
        extra_result = []
    else:
        extra_result = []
        print('L_outcome: {:.4f} (+/-) {:.4f}, L_norm: {:.4f} (+/-) {:.4f}, L_supervised: {:.4f} (+/-) {:.4f}'.format(
            np.mean(outcome_loss),std_ste(outcome_loss,std), np.mean(norm_losses),std_ste(norm_losses,std), np.mean(supervised_losses),std_ste(supervised_losses,std)), flush=True)

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
        loss_all = []
        outcome_losses,norm_losses,supervised_losses = [],[],[]
        actions, rewards, lengths, states = [],[],[],[]

        for obs,state,action,avail_action,reward,terminated,filled,length in dataloader:
            inputs = [obs,state,action,avail_action,reward,terminated,filled,length]
            outcome_loss,norm_loss,supervised_loss, qs, action, reward, length = model_computation(
                inputs,model,args,use_cuda=eval_use_cuda,train=False)

            if False: # n_optim == 1:
                if args.DIL:
                    loss = args.lmd_TD * outcome_loss + args.lmd * norm_loss
                elif "DQN" in args.model:
                    loss = args.lmd_TD * outcome_loss + args.lmd * norm_loss + args.lmd2 * supervised_loss
                else: 
                    loss = args.lmd_TD * outcome_loss
                outcome_losses.append(np.array(outcome_loss.item())[np.newaxis])
                loss_all.append(np.array(loss)[np.newaxis])
                norm_losses.append(np.array(norm_loss.item())[np.newaxis])
                supervised_losses.append(np.array(supervised_loss.item())[np.newaxis])
            else:
                loss = [[] for _ in range(n_optim)]
                for n in range(n_optim):
                    if args.DIL:
                        loss[n] = args.lmd_TD * outcome_loss[n] + args.lmd * norm_loss[n]
                    elif "DQN" in args.model:
                        loss[n] = args.lmd_TD * outcome_loss[n] + args.lmd * norm_loss[n] + args.lmd2 * supervised_loss[n]
                    else: 
                        loss[n] = args.lmd_TD * outcome_loss[n]

                outcome_losses.append(torch.stack(outcome_loss).to('cpu').detach().numpy().copy())
                loss_all.append(torch.stack(loss).to('cpu').detach().numpy().copy())
                norm_losses.append(torch.stack(norm_loss).to('cpu').detach().numpy().copy())
                supervised_losses.append(torch.stack(supervised_loss).to('cpu').detach().numpy().copy())
            # detach
            qs = detach(qs,eval_use_cuda)

            action = detach(action,eval_use_cuda)
            reward = detach(reward,eval_use_cuda)
            length = detach(length,eval_use_cuda)

            # append            
            len_time = qs.shape[1]
            Qs.append(qs.transpose((0,2,1,3))) # batch,agents,time,actions
            #try: Qs.append(qs.transpose((1,0,2)).reshape((-1,args.n_agents,len_time,args.n_actions))) # batch,agents,time,actions
            #except: import pdb; pdb.set_trace()
            actions.append(action) # batch,agents,time,actions
            rewards.append(reward)
            states.append(state)
            lengths.append(length)
            
        try: display_loss(outcome_losses, norm_losses, supervised_losses, epoch, args)
        except: import pdb; pdb.set_trace()
        # concatenate
        try: Qs = np.concatenate(Qs)
        except: import pdb; pdb.set_trace()
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        states = np.concatenate(states)
        lengths = np.concatenate(lengths)
        loss_all = np.concatenate(loss_all)
        outcome_losses = np.concatenate(outcome_losses)
        norm_losses = np.concatenate(norm_losses)
        supervised_losses = np.concatenate(supervised_losses)

        # for saving
        outcomes = [Qs, outcome_losses, norm_losses, supervised_losses, loss_all]
        inputs = [states, actions, rewards, lengths]
        return outcomes, inputs

def detach(data,eval_use_cuda):
    if eval_use_cuda:
        return data.to('cpu').detach().data.numpy()
    else:
        return data.detach().data.numpy()

def load_best_model(save_model,args,model, optimizer):
    n_optim = 1 if not 'animarl' in model.env_name else args.n_all_agents
    # weight_before = model.fc_advantage[0][0].weight.clone()
    model.load_state_dict(torch.load("{}agent.th".format(save_model), map_location=lambda storage, loc: storage))
    # weight_after = model.fc_advantage[0][0].weight
    #if torch.sum(torch.abs(weight_before-weight_after)) < 1e-6: 
    #    import pdb; pdb.set_trace()
    if False: # n_optim == 1:
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
    elif args.env == "animarl_silkmoth":
        args.n_agents = 1
        args.n_enemies = 0
        args.reward = 'reach'
    n_agents = args.n_agents
    seed = 0
    CUDA = True
    args.behavior = False 
    args.transition = False

    # parameters
    n_actions = args.n_actions
    val_devide = args.val_devide

    args.n_all_agents = args.n_agents + args.n_enemies 
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
        # input_data_path = args.data_path # '/home/fujii/workspace3/work/tag/analysis/cae/npy/'  
        args.state_dim = 4
        args.obs_shape = 19
        n_files = 8
        
        data_loader = data_loader_agent_2vs1
        all_indices = np.arange(n_files*100)

        if True: # args.option == 'CF': 
            preprocessed = np.load(os.path.join(args.input_data_path,'agent'+ '_'+ str(args.trainSize)+'_CF.npz'), allow_pickle=True)
            args.max_len = 148
        else: 
            preprocessed = np.load(os.path.join(args.input_data_path,'agent'+ '_'+ str(args.trainSize)+'.npz'), allow_pickle=True)
            args.max_len = 42 # 302
 
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
        if False:
            for j in range(n_files):
                filename = "pos_val_rep_2on1_indiv_K_" + str(j) + ".npz"
                rep = np.load(os.path.join(args.data_path, "agent", filename), allow_pickle=True) 
                reward_list = np.array(rep['rewards']).squeeze()
                rewards,lengths = [],[]
                for i in range(100):
                    rewards.append(np.array(reward_list[i][-1]))
                    lengths.append(np.array(reward_list[i]).shape[0])
                rewards = np.array(rewards)
                lengths = np.array(lengths)
                # print(rewards)
                # print(np.max(lengths))
                # print(np.sum(lengths))
    elif args.env == "animarl_fly_2vs1":
        args.state_dim = 4
        args.obs_shape = 19
        n_files = 107#114
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
        args.max_len = 501
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
            args.origin = np.array([0,0])
        # states = preprocessed['arr_0']
        args.n_folders = 10
        args.n_files = 6 # states.shape[0]
        all_indices = np.arange(args.n_files*args.n_folders)
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
    
    args.data = args.env
    args.state_shape = args.n_all_agents*args.state_dim
    n_feature = args.obs_shape + n_actions

    lengths = preprocessed['arr_3']

                
    # train/valid/test split
    if args.env == "animarl_agent_2vs1":
        if True: # args.option == 'CF':
            train_ids = np.concatenate([all_indices[:200],all_indices[379:579]])
            val_ids = np.concatenate([all_indices[200:225],all_indices[579:604]])
            # short_1 = all_indices[286:370][lengths[286:370] < 90][:25]　
            # short_2 = all_indices[663:747][lengths[663:747] < 90][:25]　
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
        if args.trainSize < 240:
            indices = np.linspace(0, 240 - 1, args.trainSize, dtype=int)
            train_ids = all_indices[indices]
        else:        
            train_ids = all_indices[:args.trainSize]
        val_ids = all_indices[240:260]
        test_ids = all_indices[260:280]
    elif args.test_cond is None and "animarl_silkmoth" in args.env:
        test_ids = np.concatenate([all_indices[18:30],all_indices[48:60]],0) # movie 4-5
        train_val_ids = np.concatenate([all_indices[:18],all_indices[30:48]],0) # movie 1-3
        # test_ids = np.concatenate([all_indices[24:30],all_indices[54:60]],0) # movie 5
        # train_val_ids = np.concatenate([all_indices[:24],all_indices[30:54]],0) # movie 1-4
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
    if 'DQN' in args.model:
        args.model_variant = args.model
        args.dueling = True
        if 'DQN_RNN' in args.model:
            args.rnn_hidden_dim = args.hidden_size
            model = RNNAgent(n_feature, args)
        else:
            model = MLPAgent(n_feature, args)
    elif 'RNN' in args.model:
        args.dueling = False
        model = RNNAgent(n_feature, args)
        args.model_variant = args.model # .replace('RNN_','')

    params = list(model.parameters())

    if not "animarl" in model.env_name:
        optimizer = optim.Adam(model.params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = [optim.Adam(model.params[n], lr=args.lr, weight_decay=args.weight_decay) for n in range(args.n_all_agents)]

    # same model
    func = 'Q'
    # str_param = ''
    str_param = '_'+args.reward # score' if args.score_only else '_checkpoints'
    str_param += '_AS' if args.AS else ''
    str_param += '_DIL' if args.DIL else ''
    str_param += '_'+args.option if args.option is not None else ''

    args.save_model = os.path.join(args.result_path, 'model_'+func, args.model+str_param+'-'+args.data+'-ts-'+str(args.trainSize)+'-lmd2-'+str(args.lmd2)+'-lr-'+str(args.lr)) + os.sep
    args.save_results = os.path.join(args.result_path, 'results_'+func, args.model+str_param+'-'+args.data+'-ts-'+str(args.trainSize)+'-lmd2-'+str(args.lmd2)+'-lr-'+str(args.lr)) + os.sep
    # args.save_model = '../AniMARL_results/model_'+func+'/'+args.model+str_param+'-'+args.data+'/'
    # args.save_results = '../AniMARL_results/results_'+func+'/'+args.model+str_param+'-'+args.data+'/'
    if not os.path.isdir(args.save_model): # setting.OUTPUT_PATH+'/model_Q'):
        os.makedirs(args.save_model)
    if not os.path.isdir(args.save_results):
        os.makedirs(args.save_results)
    print("save_model ==> ", args.save_model, 'agent.th')
    print("save_results ==> ", args.save_results)
    
    if CUDA: # use_cuda:
        print("====> Using CUDA device: ", torch.cuda.current_device(), flush=True)
        for n in range(args.n_all_agents):
            if 'RNN' in args.model:
                model.rnn[n].cuda() # rnn
            model.fc_in[n].cuda() # fc1    
            if args.dueling:
                model.fc_state[n].cuda() # 
                model.fc_advantage[n].cuda() #
            else:
                model.fc_out[n].cuda() #
            if args.option == 'CF':
                model.fc_out_cond[n].cuda()

        args.total_params = 0
        for param in model.parameters():
            args.total_params += param.numel()/args.n_all_agents

    if args.cont:
        print('args.cont = True')
        if os.path.exists(args.save_model+'agent.th'): # .pt'): 
            model, optimizer = load_best_model(args.save_model,args,model,optimizer)
            print('best model was loaded')   
        else:
            print('args.cont = True but file did not exist')
            
    # training and test
    trainmodel(train_loader, val_loader, test_loader, model, args, epochs= args.epochs,
                  criterion=F.mse_loss, optimizer=optimizer,
                  use_cuda=CUDA, save_model=args.save_model, TEST=args.TEST)
