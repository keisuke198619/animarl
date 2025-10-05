from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        self.env_name = args.env_args['env_name']
        self.action_selector = action_REGISTRY[args.action_selector](args) # "epsilon_greedy"

        if self.args.action_selector == 'direct':
            self.actor_indiv = args.actor_indiv

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        # print(chosen_actions)
        return chosen_actions, agent_outputs

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.action_selector == 'direct':
            sample = True if not test_mode else False
            if "rnn" in self.args.agent or "sac" in self.args.agent:
                agent_outs, self.hidden_states = self.agent.act(agent_inputs, hidden_states=self.hidden_states, sample=sample)
            else:
                agent_outs, _ = self.agent.act(agent_inputs,sample=sample)
            agent_outs = th.stack(agent_outs)
        else: 
            agent_outs, self.hidden_states, _ = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits": # default: "q"

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        try: return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        except: import pdb; pdb.set_trace()

    def init_hidden(self, batch_size):
        if "rnn" in self.args.agent or "sac" in self.args.agent:
            self.hidden_states = []
            for n in range(self.n_agents):
                hidden_states = th.zeros(self.args.hidden_depth, 1, self.args.hidden_dim)
                self.hidden_states.append(hidden_states)
                # self.hidden_states.append(self.agent.init_hidden(n).unsqueeze(0).expand(batch_size, 1, -1))
            self.hidden_states = th.stack(self.hidden_states,dim=1).to(self.args.GPU)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.to(self.args.GPU)

    def save_models(self, path):
        if self.args.action_selector == 'direct':
            try: 
                th.save(self.agent.actor.state_dict(), "{}/actor.th".format(path))
                th.save(self.agent.critic.state_dict(), "{}/critic.th".format(path))
                if self.actor_indiv:
                    for n in range(self.n_agents):
                        th.save(self.agent.actor_optimizer[n].state_dict(), "{}/actor_optimizer_{}.th".format(path,str(n)))
                        th.save(self.agent.critic_optimizer[n].state_dict(), "{}/critic_optimizer_{}.th".format(path,str(n)))
                        th.save(self.agent.log_alpha_optimizer[n].state_dict(), "{}/log_alpha_optimizer_{}.th".format(path,str(n)))
                elif not self.actor_indiv:
                    th.save(self.agent.actor_optimizer.state_dict(), "{}/actor_optimizer.th".format(path))
                    th.save(self.agent.critic_optimizer.state_dict(), "{}/critic_optimizer.th".format(path))
                    th.save(self.agent.log_alpha_optimizer.state_dict(), "{}/log_alpha_optimizer.th".format(path))
            except: import pdb; pdb.set_trace()
            
        else: 
            th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        # import pdb; pdb.set_trace() # self.agent.fc_common[0][0].weight　self.agent.fc1.weight
        # weight_before = self.agent.fc_advantage[0][0].weight.clone()
        if self.args.action_selector == 'direct':
            self.agent.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
            self.agent.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
            if self.actor_indiv:
                for n in range(self.n_agents):
                    self.agent.actor_optimizer[n].load_state_dict(th.load("{}/actor_optimizer_{}.th".format(path,str(n)), map_location=lambda storage, loc: storage))
                    self.agent.critic_optimizer[n].load_state_dict(th.load("{}/critic_optimizer_{}.th".format(path,str(n)), map_location=lambda storage, loc: storage))
                    self.agent.log_alpha_optimizer[n].load_state_dict(th.load("{}/log_alpha_optimizer_{}.th".format(path,str(n)), map_location=lambda storage, loc: storage))
            elif not self.actor_indiv:
                self.agent.actor_optimizer.load_state_dict(th.load("{}/actor_optimizer.th".format(path), map_location=lambda storage, loc: storage))
                self.agent.critic_optimizer.load_state_dict(th.load("{}/critic_optimizer.th".format(path), map_location=lambda storage, loc: storage))
                self.agent.log_alpha_optimizer.load_state_dict(th.load("{}/log_alpha_optimizer.th".format(path), map_location=lambda storage, loc: storage))

        else:
            print("model path: "+path)
            self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        # weight_after = self.agent.fc_advantage[0][0].weight
        # import pdb; pdb.set_trace()
        # if th.sum(th.abs(weight_before-weight_after)) < 1e-6: 
        #    import pdb; pdb.set_trace()
    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(
                0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1)
                        for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += self.args.n_actions # scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
