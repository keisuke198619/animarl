import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.env_name = args.env
        self.n_agents = args.n_all_agents
        self.hidden_size = args.rnn_hidden_dim
        self.model_variant = args.model_variant
        self.n_layers = 1

        self.fc_in = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)]) 
        self.rnn = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])  
        self.fc_out = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])  

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
            self.fc_out[n] = nn.Sequential(
                nn.Linear(self.final_hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, self.args.n_actions)
            )

        self.params = [[] for n in range(self.n_agents)]
        
        for n in range(self.n_agents):
            self.params[n] = []
            self.params[n] += list(self.fc_in[n].parameters())
            self.params[n] += list(self.rnn[n].parameters())
            self.params[n] += list(self.fc_out[n].parameters())

    def forward(self, obs, action):
        torch.autograd.set_detect_anomaly(True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchSize = obs.size(0)
        n_agents = obs.size(1)
        len_time = obs.size(2) # -1
        n_feature = obs.size(3) 
        n_actions = action.size(3)
        # inputs = obs
        inputs = torch.cat([obs,action],dim=3) # batch,agants,time,dim
        hidden_state = [torch.randn(self.n_layers,batchSize, self.hidden_size).to(device) for _ in range(n_agents)]

        action_out = [] 
        
        n=0
        for n in range(self.n_agents):
            action_out_ = []
            hidden_state_ = hidden_state[n].clone()
            for i in range(len_time): # time
                try: 
                    inputs_ = inputs[:,n,i].clone()
                    x = F.relu(self.fc_in[n](inputs_))
                except: import pdb; pdb.set_trace()
                x = x.unsqueeze(1)
                # hidden_state_ = hidden_state[n].clone()
                gru_out_, hidden_state_ = self.rnn[n](x, hidden_state_)# hidden_state[n])
                if 'animarl_agent' in self.env_name or 'animarl_silkmoth' in self.env_name:   
                    gru_out__ = torch.cat([gru_out_,inputs_[:,None,-1,None]],dim=2)
                else:
                    gru_out__ = gru_out_
                # hidden_state[n] = hidden_state_.clone()
                action_out__ = self.fc_out[n](gru_out__) # batch,1,h_dim -> batch,1,actions
                action_out_.append(action_out__)  
            
            # q_ = torch.stack(q_,dim=1) # agents,batch,actions 
            action_out_ = torch.stack(action_out_,dim=0) # time,batch,1,actions 
            action_out.append(action_out_)
        
        action_out = torch.stack(action_out, dim=2).squeeze(3) # time,batch,agents,actions 
        action_out = action_out.reshape(-1,batchSize,n_agents,n_actions).permute(1,0,2,3) # batch,time,agents,actions
    
        return action_out
