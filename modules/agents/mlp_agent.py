from shlex import quote
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_all_agents = args.n_all_agents
        self.env_name = args.env_args['env_name']
        if not "animarl" in self.env_name:
            print('not animarl in self.env_name')
            import pdb; pdb.set_trace()
        else:
            self.fc_in = nn.ModuleList([nn.ModuleList() for i in range(self.n_all_agents)])
            self.fc_state = nn.ModuleList([nn.ModuleList() for i in range(self.n_all_agents)])  
            self.fc_advantage = nn.ModuleList([nn.ModuleList() for i in range(self.n_all_agents)])    
            if 'CF' in args.cond:
                self.fc_out_cond = nn.ModuleList([nn.ModuleList() for i in range(self.n_agents)])
                self.final_hidden_size = 64+1
            else:
                self.final_hidden_size = 64
                
            for n in range(self.n_all_agents):
                self.fc_in[n] = nn.Sequential(
                    nn.Linear(input_shape, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU()
                )

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
            if 'CF' in args.cond:
                self.fc_out_cond[n] = nn.Sequential(GradientReversal(),
                    nn.Linear(self.rnn_hidden_dim, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2)
                )
            self.params = [[] for n in range(args.n_all_agents)]

            for n in range(args.n_all_agents):
                self.params[n] = []
                self.params[n] += list(self.fc_in[n].parameters())
                self.params[n] += list(self.fc_state[n].parameters())
                self.params[n] += list(self.fc_advantage[n].parameters())
                if 'CF' in args.cond:     
                    self.params[n] += list(self.fc_out_cond[n].parameters())

    def forward(self, inputs, hidden_state):
        torch.autograd.set_detect_anomaly(True)
        input_shape = inputs.shape

        if len(input_shape) == 2:
            if not "animarl" in self.env_name:
                print('not animarl in self.env_name')
                import pdb; pdb.set_trace()
            else:
                local_q = None
                gru_out = None
                q = []
                    
                for n in range(self.n_agents):
                    feature = self.fc_in[n](inputs[n]) # [hidden]
                    # feature = feature.view(feature.size(0), -1) # [hidden,1]

                    state_values = self.fc_state[n](feature)
                    advantage = self.fc_advantage[n](feature)

                    action_values = state_values + advantage - torch.mean(advantage, dim=0) # , keepdim=True

                    q.append(action_values.unsqueeze(0))
                    if 'CF' in self.args.cond:
                        try: out_cond_.append(self.fc_out_cond[n](gru_out_))  
                        except: import pdb; pdb.set_trace()  
                q = torch.stack(q)

        elif len(input_shape) == 4: # batch,agent,time,feat
            if not "animarl" in self.env_name:
                print('not animarl in self.env_name')
                import pdb; pdb.set_trace()
            else:
                local_q = None
                gru_out = None
                q = []
                inputs = inputs.permute(2,0,1,3).reshape(inputs.shape[-3], -1, inputs.shape[-1])
                for n in range(self.n_agents):
                    feature = self.fc_in[n](inputs[n]) # [-1, hidden]
                    # feature_ = feature.clone() # contiguous().view(feature.size(0), -1)

                    state_values = self.fc_state[n](feature) # [-1,1]
                    advantage_ = self.fc_advantage[n](feature) # [-1,actions]

                    action_values = state_values + advantage_ - torch.mean(advantage_, dim=1, keepdim=True) # [-1,actions]

                    action_values = action_values.reshape(input_shape[0], -1, action_values.shape[-1]) # 4448,13->32,139,13

                    q.append(action_values)
                    # del feature, state_values, advantage, action_values
                    if 'CF' in self.args.cond:
                        try: out_cond_.append(self.fc_out_cond[n](gru_out_))  
                        except: import pdb; pdb.set_trace()  
                q = torch.stack(q, dim=2) # ->32,139,4,19
        # import pdb; pdb.set_trace() # q_ = q
        return q, gru_out, local_q
