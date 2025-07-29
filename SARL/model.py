import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal
from copy import deepcopy, copy

################################## set device ##################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class small_size(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(small_size, self).__init__()

        # self.resnet = resnet18(pretrained=True).float()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def __repr__(self):
        return "small_size"

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, inp):
        inp = inp.cuda().float()
        return self.act(inp)#raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def get_weights(self):
        return copy({k: v.cpu() for k, v in self.state_dict().items()})

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else copy(p.grad.data.cpu().numpy())
            grads.append(copy(grad))
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                #try:
                #    print(len(g), len(g[0]), p.shape)
                #except:
                #    print(len(g), p.shape)
                p.grad = torch.cuda.FloatTensor(g)#from_numpy(g)



class ActorCriticConv(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCriticConv, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device= "cuda"
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)#.half()
        
        self.first_conv = nn.Sequential(
                nn.Conv2d(int(state_dim//(1024*256)), 1, (7,7), (3, 3)),
                nn.Tanh(),
                nn.Conv2d(1, 1, (7, 7), (3,3)),
                nn.Tanh(),
                )

        # actor
        if has_continuous_action_space:

            self.actor = nn.Sequential(
                nn.Linear(2912, 64),
                #nn.Tanh(),
                #nn.Linear(3, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(2912, 64),
                #nn.Tanh(),
                #nn.Linear(3, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(2912, 64),
            #nn.Tanh(),
            #nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = self.first_conv(state)
        state = torch.flatten(state.to(self.device))
        if self.has_continuous_action_space:
            #state_conv = self.first_conv(state)
            #state_conv = torch.flatten(state_conv).unsqueeze(0)#.to(device)
            action_mean = self.actor(state).float().to(self.device)#_conv).float().to(self.device)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).float().to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            #state_conv = self.first_conv(state)
            #state_conv = torch.flatten(state_conv).unsqueeze(0)
            action_probs = self.actor(state)#_conv)
            dist = Categorical(action_probs)
        #print(action_mean, dist)

        action = dist.sample().to(self.device)
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        state = self.first_conv(state)
        state = torch.flatten(state.to(self.device), start_dim=1)
        if self.has_continuous_action_space:
            #state_conv = self.first_conv(state)
            #state_conv = torch.flatten(state_conv).unsqueeze(0).to(self.device)
            action_mean = self.actor(state).float().to(self.device)#_conv).float().to(self.device)

            action_var = self.action_var.expand_as(action_mean).float()
            cov_mat = torch.diag_embed(action_var).to(device).float().to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            #state_conv = self.first_conv(state)
            #state_conv = torch.flatten(state).unsqueeze(0)#_conv).unsqueeze(0)
            action_probs = self.actor(state)#_conv)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action).to(self.device)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)#_conv)

        return action_logprobs, state_values, dist_entropy

    def get_weights(self):
        return copy({k: v.cpu() for k, v in self.state_dict().items()})

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else copy(p.grad.data.cpu().numpy())
            grads.append(copy(grad))
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.cuda.FloatTensor(g)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device= "cuda"
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)#.half()
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = state.to(self.device)
        if self.has_continuous_action_space:
            action_mean = self.actor(state).float()
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).float()
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        state = state.to(self.device)
        if self.has_continuous_action_space:
            action_mean = self.actor(state).float()

            action_var = self.action_var.expand_as(action_mean).float()
            cov_mat = torch.diag_embed(action_var).to(device).float()
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def get_weights(self):
        return copy({k: v.cpu() for k, v in self.state_dict().items()})

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else copy(p.grad.data.cpu().numpy())
            grads.append(copy(grad))
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class PPOLSTMModel(torch.nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(PPOLSTMModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.has_continuous_action_space = args.continuous
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.lstm = nn.LSTMCell(108, self.hidden_size)
        num_outputs = action_space

        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)

        self.cov_var = torch.full(size=(num_outputs,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        if self.has_continuous_action_space:
            relu_gain = nn.init.calculate_gain("tanh")
        else:
            relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2.bias.data.fill_(0)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv3.bias.data.fill_(0)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

    def forward(self, inputs, hx, cx, actval):
        if self.has_continuous_action_space:
            activ = F.tanh
        else:
            activ = F.relu
        x = activ(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = activ(F.max_pool2d(self.conv2(x), 2, 2))
        x = activ(F.max_pool2d(self.conv3(x), 2, 2))
        x = activ(F.max_pool2d(self.conv4(x), 2, 2))

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        if actval:
            return self.actor_linear(x), hx, cx
        return self.critic_linear(x), self.actor_linear(x), hx, cx

    def act(self, state, hx, cx):
        if self.has_continuous_action_space:
            action_mean, hx, cx = self.forward(state, hx, cx, True)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs, hx, cx = self.forward(state, hx, cx, True)
            action_probs = F.softmax(action_probs, dim=-1)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), hx, cx

    def evaluate(self, state, action, hx, cx):
        if self.has_continuous_action_space:
            state_value, action_probs, hx, cx = self.forward(state, hx, cx, False)
            action_var = self.action_var.expand_as(action_probs)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_probs.float(), cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            state_value, action_probs, hx, cx = self.forward(state, hx, cx, False)
            action_probs = F.softmax(action_probs, dim=-1)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy, hx, cx

    def get_weights(self):
        return copy({k: v.cpu() for k, v in self.state_dict().items()})

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else copy(p.grad.data.cpu().numpy())
            grads.append(copy(grad))
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
