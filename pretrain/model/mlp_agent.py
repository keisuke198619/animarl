import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_chase import GradientReversal

class MLPAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.env_name = args.env
        self.n_agents = args.n_all_agents
        self.model_variant = args.model_variant
        if not "animarl" in self.env_name:
            print('not animarl in self.env_name')
            import pdb; pdb.set_trace()
            self.params = list(self.parameters())
        else:
            self.fc_in = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])
            self.fc_state = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])  
            self.fc_advantage = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])   
            if self.args.option == 'CF' or self.args.option == 'CF2':
                self.fc_out_cond = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)]) 
            # self.fc_in,self.fc_state,self.fc_advantage = [],[],[]
            for n in range(self.n_agents):
                self.fc_in[n] = nn.Sequential(
                    nn.Linear(input_shape, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU()
                )

                self.fc_state[n] = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )

                self.fc_advantage[n] = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, args.n_actions)
                )
                if self.args.option == 'CF' or self.args.option == 'CF2':
                    self.fc_out_cond[n] = nn.Sequential(GradientReversal(),
                        nn.Linear(64, 8),
                        nn.ReLU(),
                        nn.Linear(8, 2)
                    )
            self.params = [[] for n in range(self.n_agents)]
            
            for n in range(self.n_agents):
                self.params[n] = []
                self.params[n] += list(self.fc_in[n].parameters())
                self.params[n] += list(self.fc_state[n].parameters())
                self.params[n] += list(self.fc_advantage[n].parameters())
                if self.args.option == 'CF' or self.args.option == 'CF2':   
                    self.params[n] += list(self.fc_out_cond[n].parameters())
    def forward(self, obs, action):
        torch.autograd.set_detect_anomaly(True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchSize = obs.size(0)
        n_agents = obs.size(1)
        len_time = obs.size(2) # -1
        n_feature = obs.size(3)+action.size(3)
        n_actions = action.size(3)

        if not "animarl" in self.env_name:
            action_ = torch.cat([torch.zeros(batchSize,n_agents,1,n_actions).to(device),action[:,:,:-1]],2)
            inputs = torch.cat([obs,action],dim=3).reshape(-1,len_time,n_feature) # [:,:,1:] [:,:,:-1]
        else:
            try: action_ = torch.cat([torch.zeros(batchSize,n_agents,1,n_actions).to(device),action[:,:,:-1]],2)
            except: import pdb; pdb.set_trace()
            inputs = torch.cat([obs,action_],dim=3) # batch,agants,time,dim
            
        qs,out_conds = [],[]
        
        if not "animarl" in self.env_name:
            print('not animarl in self.env_name')
            import pdb; pdb.set_trace()
        else:
            n=0
            for n in range(self.n_agents):
                q_,out_cond_ = [],[]
                for i in range(len_time): # time
                    inputs_ = inputs[:,n,i].clone()
                    feature = self.fc_in[n](inputs_) # [hidden]
                    # feature = feature.view(feature.size(0), -1) # [hidden,1]

                    state_values = self.fc_state[n](feature)
                    advantage = self.fc_advantage[n](feature)

                    action_values = state_values + advantage - torch.mean(advantage, dim=0) # , keepdim=True

                    q_.append(action_values) # x is OK # q__) #  gru_out_
                    if self.args.option == 'CF' or self.args.option == 'CF2':
                        out_cond_.append(self.fc_out_cond[n](feature))
                # q_ = torch.stack(q_,dim=1) # agents,batch,actions 
                q_ = torch.stack(q_,dim=0) # time,batch,1,actions 
                qs.append(q_)
                if self.args.option == 'CF' or self.args.option == 'CF2':
                    out_cond_ = torch.stack(out_cond_,dim=0) # time,batch,1,actions 
                    out_conds.append(out_cond_)
            qs = torch.stack(qs, dim=2).squeeze(3) # time,batch,agents,actions 
        qs = qs.reshape(-1,batchSize,n_agents,n_actions).permute(1,0,2,3) # batch,time,agents,actions
        if self.args.option == 'CF' or self.args.option == 'CF2':
            out_conds = torch.stack(out_conds, dim=2).squeeze(3).permute(1,2,0,3) # batch,agents,time,2
            
        return qs, out_conds