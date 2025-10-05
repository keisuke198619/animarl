import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_chase import GradientReversal

class RNNAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.env_name = args.env
        self.n_agents = args.n_all_agents
        self.n_actions = args.n_actions
        self.hidden_size = args.rnn_hidden_dim
        self.model_variant = args.model_variant
        self.n_layers = 1
        self.rnn = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])      
        self.fc_in = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)]) 
        if args.dueling:
            self.fc_state = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])  
            self.fc_advantage = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])    
        else:
            self.fc_out = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])  
        if self.args.option is not None and 'CF' in self.args.option:
            self.fc_out_cond = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])

        if 'animarl_agent' in self.env_name or 'animarl_silkmoth' in self.env_name:   
            self.final_hidden_size = self.hidden_size+1
        else:
            self.final_hidden_size = self.hidden_size

        for n in range(self.n_agents):
            self.fc_in[n] = nn.Sequential(
                    nn.Linear(input_shape, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU()
                )
            self.rnn[n] = nn.GRU(
                input_size=self.hidden_size,
                num_layers=1,
                hidden_size=self.hidden_size,
                batch_first=True,
                )
            if args.dueling:
                self.fc_state[n] = nn.Sequential(
                    nn.Linear(self.final_hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )

                self.fc_advantage[n] = nn.Sequential(
                    nn.Linear(self.final_hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.n_actions)
                )
            else:    
                self.fc_out[n] = nn.Sequential(
                    nn.Linear(self.final_hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.n_actions)
                )
            if self.args.option is not None and 'CF' in self.args.option:
                self.fc_out_cond[n] = nn.Sequential(GradientReversal(),
                    nn.Linear(self.hidden_size, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2)
                )
        self.params = [[] for n in range(self.n_agents)]
            
        for n in range(self.n_agents):
            self.params[n] = []
            self.params[n] += list(self.fc_in[n].parameters())
            self.params[n] += list(self.rnn[n].parameters())
            if args.dueling:
                self.params[n] += list(self.fc_state[n].parameters())
                self.params[n] += list(self.fc_advantage[n].parameters())
            else:
                self.params[n] += list(self.fc_out[n].parameters())
            if self.args.option is not None and 'CF' in self.args.option: 
                self.params[n] += list(self.fc_out_cond[n].parameters())

    def forward(self, obs, action):
        torch.autograd.set_detect_anomaly(True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchSize = obs.size(0)
        n_agents = obs.size(1)
        len_time = obs.size(2) # -1
        n_feature = obs.size(3)+action.size(3)
        n_actions = action.size(3)

        inputs = torch.cat([obs,action],dim=3) # batch,agants,time,dim
        hidden_state = [torch.randn(self.n_layers,batchSize, self.hidden_size).to(device) for _ in range(n_agents)]

        qs,out_conds = [],[]
        
        n=0
        for n in range(self.n_agents):
            q_,out_cond_ = [],[]
            hidden_state_ = hidden_state[n].clone()
            for i in range(len_time): # time
                inputs_ = inputs[:,n,i].clone()
                x = F.relu(self.fc_in[n](inputs_))
                x = x.unsqueeze(1)
                # hidden_state_ = hidden_state[n].clone()
                gru_out_, hidden_state_ = self.rnn[n](x, hidden_state_)# hidden_state[n])
                # hidden_state[n] = hidden_state_.clone()
                if self.args.option is not None and 'CF' in self.args.option:
                    out_cond_.append(self.fc_out_cond[n](gru_out_))

                if 'animarl_agent' in self.env_name or 'animarl_silkmoth' in self.env_name:   
                    gru_out__ = torch.cat([gru_out_,inputs_[:,None,-1,None]],dim=2)
                else:
                    gru_out__ = gru_out_
                    
                if self.args.dueling:
                    state_values = self.fc_state[n](gru_out__)
                    advantage = self.fc_advantage[n](gru_out__)
                    q__ = state_values + advantage - torch.mean(advantage, dim=0) # , keepdim=True
                else:
                    q__ = self.fc_out[n](gru_out__) # batch,1,h_dim -> batch,1,actions

                q_.append(q__) # x[:,:,:13].clone()) # x is OK # q__) #  gru_out_
            
            # q_ = torch.stack(q_,dim=1) # agents,batch,actions 
            q_ = torch.stack(q_,dim=0) # time,batch,1,actions 
            qs.append(q_)
            if self.args.option is not None and 'CF' in self.args.option:
                out_cond_ = torch.stack(out_cond_,dim=0) # time,batch,1,actions 
                out_conds.append(out_cond_)
        qs = torch.stack(qs, dim=2).squeeze(3) # time,batch,agents,actions 
        qs = qs.reshape(-1,batchSize,n_agents,n_actions).permute(1,0,2,3) # batch,time,agents,actions
        if self.args.option is not None and 'CF' in self.args.option:
            out_conds = torch.stack(out_conds, dim=2).squeeze(3).permute(1,2,0,3) # batch,agents,time,2
            
        return qs, out_conds
