from shlex import quote
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_chase import GradientReversal

class RNNAgent(nn.Module):

    def __init__(self, input_shape, args, n_agents=0, env_name='',rnn_hidden_dim=0,n_actions=0,mixer='None'):
        super(RNNAgent, self).__init__()
        if args is not None:
            self.args = args
            self.n_agents = args.n_agents
            self.env_name = args.env_args['env_name']
            self.rnn_hidden_dim = args.rnn_hidden_dim
            self.mixer = args.mixer
            self.n_actions = args.n_actions
        else:
            self.n_agents = n_agents
            self.env_name = env_name
            self.rnn_hidden_dim = rnn_hidden_dim
            self.mixer = mixer
            self.n_actions = n_actions

        self.rnn = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])     
        self.fc_in = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)]) 
        if args.dueling:
            self.fc_state = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])  
            self.fc_advantage = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])    
        else:
            self.fc_out = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])    
        if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':
            self.fc_out_cond = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])

        if 'animarl_agent' in self.env_name or 'animarl_silkmoth' in self.env_name:   
            self.final_hidden_size = self.rnn_hidden_dim+1
        else:
            self.final_hidden_size = self.rnn_hidden_dim

        for n in range(self.n_agents):
            self.fc_in[n] = nn.Sequential(
                    nn.Linear(input_shape, self.rnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim),
                    nn.ReLU()
                )
            self.rnn[n] = nn.GRU(
                input_size=self.rnn_hidden_dim,
                num_layers=1,
                hidden_size=self.rnn_hidden_dim,
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
                    nn.Linear(32, args.n_actions)
                )
            else:    
                self.fc_out[n] =  nn.Sequential(
                    nn.Linear(self.final_hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, args.n_actions)
                )
            if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':
                self.fc_out_cond[n] = nn.Sequential(GradientReversal(),
                    nn.Linear(self.rnn_hidden_dim, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2)
                ).to(args.device)
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
            if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':     
                self.params[n] += list(self.fc_out_cond[n].parameters())

    def forward(self, inputs, hidden_state):
        try: 
            if len(hidden_state.shape) == 2:
                hidden_state = hidden_state.unsqueeze(0)
        except: import pdb; pdb.set_trace()

        hidden_state = hidden_state.contiguous()
        input_shape = inputs.shape
        #if self.mixer == 'None':
        local_q = None

        if len(input_shape) == 2:
            gru_out,q,out_cond = [],[],[]

            if len(hidden_state.shape) == 4:
                if hidden_state.shape[0] == 1:
                    hidden_state = hidden_state.squeeze(0).contiguous() # 1,3,1,64->3,1,64
                elif hidden_state.shape[2] == 1:
                    hidden_state = hidden_state.squeeze(2).permute(1,0,2).contiguous()
                else:
                    import pdb; pdb.set_trace()
            else:
                hidden_state = hidden_state.permute(2,0,1).contiguous() # 1,64,3->3,1,64
                
            for n in range(self.n_agents):
                x = F.relu(self.fc_in[n](inputs[n]))
                x = x.unsqueeze(0).unsqueeze(0)
                gru_out_, _ = self.rnn[n](x, hidden_state[n].unsqueeze(0)) # 
                if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':
                    try: out_cond.append(self.fc_out_cond[n](gru_out_))  
                    except: import pdb; pdb.set_trace()

                if 'animarl_agent' in self.env_name or 'animarl_silkmoth' in self.env_name:   
                    gru_out__ = torch.cat([gru_out_, inputs[n][-1][None,None,None]], dim=2)
                else:
                    gru_out__ = gru_out_

                if self.args.dueling:
                    state_values = self.fc_state[n](gru_out__)
                    advantage = self.fc_advantage[n](gru_out__)
                    q__ = state_values + advantage - torch.mean(advantage, dim=0) # , keepdim=True
                else:
                    q__ = self.fc_out[n](gru_out__) # batch,1,h_dim -> batch,1,actions

                gru_out.append(gru_out_.squeeze())
                q.append(q__) # self.fc2[n](gru_out[n]))
            
            gru_out = torch.stack(gru_out, dim=1)
            q = torch.stack(q)
            if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':
                out_cond = torch.stack(out_cond)

        elif len(input_shape) == 4: # batch,agent,time,feat
            gru_out,q,out_cond = [],[],[]

            if len(hidden_state.shape) == 4:
                import pdb; pdb.set_trace()
                hidden_state = hidden_state.squeeze(0).contiguous() # 1,3,1,64->3,1,64????
            else:
                hidden_state = hidden_state.repeat(input_shape[0], 1, 1).contiguous() # 1,3,64 -> 32,3,64
                # hidden_state = hidden_state.reshape(inputs.shape[0],inputs.shape[1],-1).contiguous() # 1,96,64->32,3,64
            inputs = inputs.permute(2,0,1,3).reshape(inputs.shape[-3], -1, inputs.shape[-1])
            for n in range(self.n_agents):
                x = F.relu(self.fc_in[n](inputs[n]))
                x = x.reshape(-1, input_shape[2], x.shape[-1])
                gru_out_, _ = self.rnn[n](x, hidden_state[:,n].unsqueeze(0).contiguous())
                gru_out_c = gru_out_.reshape(-1, gru_out_.shape[-1])
                gru_out.append(gru_out_c)
                if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':
                    out_cond.append(self.fc_out_cond[n](gru_out_))  
                    
                if 'animarl_agent' in self.env_name or 'animarl_silkmoth' in self.env_name:   
                    gru_out_c_ = torch.cat([gru_out_c, inputs[n][:,-1][:,None]], dim=1)
                else:
                    gru_out_c_ = gru_out_c
                if self.args.dueling:
                    state_values = self.fc_state[n](gru_out_c_)
                    advantage = self.fc_advantage[n](gru_out_c_)
                    q_= state_values + advantage - torch.mean(advantage, dim=0) # , keepdim=True
                else:
                    q_ = self.fc_out[n](gru_out_c_) # batch,1,h_dim -> batch,1,actions

                q_ = q_.reshape(input_shape[0], -1, q_.shape[-1]) # 4448,13->32,139,13

                q.append(q_)
            
            gru_out = torch.stack(gru_out, dim=1)
            q = torch.stack(q, dim=2) # ->32,139,4,19
            if self.args.cond == 'CF' or self.args.cond == 'CF4' or self.args.cond == 'CF6':
                out_cond = torch.stack(out_cond, dim=2) # batch,time,agent,feat

        return q, gru_out, out_cond
